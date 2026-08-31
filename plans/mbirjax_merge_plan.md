# MACE4DModel — mbirjax Merge Plan

## 1. Parameter Classification

mbirjax assigns parameters to one of four categories: required constructor arguments,
optional constructor arguments managed via `set_params`, required arguments to `recon()`,
and optional arguments to `recon()`.  The tables below place each parameter of
`MACE4DModel` into one of these categories.

### Category 1 — Required constructor arguments

| Parameter | Type | Notes |
|-----------|------|-------|
| `model_list` | `list[ConeBeamModel]` | One model per time frame; encodes the scan geometry for that frame. |

`sino_list` moves to `recon()` as a required argument (Category 3).  This change
follows the mbirjax convention that sinogram data is passed at call time, not at
construction.  Weights are computed at the start of `recon()` instead of in `__init__`.

### Category 2 — Optional constructor arguments, managed via `set_params`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mace_prior_weight` | `0.5` | Total weight of the three prior agents.  A scalar `w` maps to weights `[1−w, w/3, w/3, w/3]` across `[fwd, xyt, yzt, xzt]`.  A three-element list `[w0, w1, w2]` sets each prior weight independently. |
| `rho_mann` | `0.5` | Mann iteration step size (ADMM ρ). |
| `max_mace_itr` | `10` | Number of outer MACE iterations. |
| `prox_num_iterations` | `3` | Maximum number of `prox_map` iterations per MACE step. |
| `prox_stop_threshold` | `0.02` | `prox_map` convergence threshold, expressed as percent change. |
| `sigma_prox` | `None` | Proximal sigma.  `None` lets mbirjax select a value automatically. |
| `dejitter` | `True` | Apply DCT-I temporal dejitter inside each agent. |
| `frames_per_rotation` | `6` | Number of time frames per full rotation.  This value sets the period of the temporal dejitter filter. |
| `weight_type` | `"transmission_root"` | Sinogram weight type passed to `gen_weights`. |
| `verbose` | `1` | Verbosity level.  `0` = silent, `1` = progress, `2` = debug.  Already registered in mbirjax's parameter system. |

### Category 3 — Required `recon()` arguments

| Parameter | Type | Notes |
|-----------|------|-------|
| `sino_list` | `list[ndarray]` | Per-frame sinograms.  Each array has shape `(views_per_frame, det_rows, det_cols)`. |

### Category 4 — Optional `recon()` arguments

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `devices` | `None` | `None` uses all visible GPUs.  An integer `n` uses the first `n` devices.  A list specifies devices explicitly. |
| `init_image` | `None` | A 4D array of shape `(nt, nx, ny, nz)` to use as the starting point.  When provided, the per-frame MBIR initialization is skipped. |
| `init_dir` | `None` | Directory for caching the computed initialization image (`init_image.npy`).  On a subsequent run with the same settings, the cached image is loaded instead of recomputed. |
| `log_dir` | `None` | Directory for `run_info.txt`, `timing_log.csv`, and `task_log.csv`.  `None` writes no log files. |

---

## 2. Open Interface Question — Where Does `construct_time_frames` Run?

The current code calls `construct_time_frames` as a separate step before constructing
`MACE4DModel`.  Moving this call inside the model is cleaner for the user, but it raises
a question about where the full sinogram `sino` enters: at construction time or at
`recon()` time.  The two options have different tradeoffs.

### Option A — Sinogram enters at construction time

```python
mace = mj.MACE4DModel(
    sino=sino,
    model=ct_model,
    frames_per_rotation=6,
    frame_overlap_factor=2.0,
    mace_prior_weight=0.5,
)
recon_4d = mace.recon(init_dir="./output/init", log_dir="./output/logs")
```

`construct_time_frames` runs inside `__init__`, producing `sino_list` and `model_list`
as internal state.  The user never calls `construct_time_frames` directly.  This is the
simplest interface and requires the fewest steps.

The tradeoff is that sinogram data enters at construction time.  This deviates from
the mbirjax convention, where the constructor receives geometry and `recon()` receives
data.

### Option B — Sinogram enters at `recon()` time

```python
mace = mj.MACE4DModel(
    model=ct_model,
    frames_per_rotation=6,
    frame_overlap_factor=2.0,
    mace_prior_weight=0.5,
)
recon_4d = mace.recon(sino, init_dir="./output/init", log_dir="./output/logs")
```

`construct_time_frames` runs inside `recon()`.  The constructor receives geometry only,
and `sino` is passed to `recon()` as its first required argument.  This matches the
mbirjax convention exactly: `ConeBeamModel.__init__` takes geometry, and
`ConeBeamModel.recon(sinogram, ...)` takes data.

The tradeoff is that `construct_time_frames` is no longer a standalone utility.  A user
who wants to inspect the per-frame sinograms or models before running the reconstruction
must call `construct_time_frames` separately, which means it still needs to be exported
as a public function.

### Comparison

| | Option A | Option B |
|---|---|---|
| Steps for the user | fewer | more |
| Matches mbirjax convention | no | yes |
| `construct_time_frames` stays public | no | yes (still needed for inspection) |
| Sinogram data in constructor | yes | no |

The parameter tables in Section 1 reflect Option B.  Under Option A, `sino` moves from
Category 3 (`recon()` required) to Category 1 (constructor required), and
`frame_overlap_factor` moves from Category 2 to Category 1 alongside `frames_per_rotation`.

---

## 3. Proposed Interface (Option B)

The interface below assumes Option B.  It follows the mbirjax convention of passing
geometry to the constructor and data to `recon()`.

```python
# Construction (geometry only)
mace = mj.MACE4DModel(model=ct_model, frames_per_rotation=6, frame_overlap_factor=2.0,
                      mace_prior_weight=0.5, max_mace_itr=10)
mace.set_params(rho_mann=0.5, dejitter=True)

# Reconstruction (sinogram data passed here)
recon_4d = mace.recon(sino, init_dir="./output/init", log_dir="./output/logs")
```

Under Option B, `construct_time_frames` remains a public module-level function so that
users can inspect per-frame sinograms and models before reconstruction.  This is
consistent with how mbirjax exposes utilities such as `copy_ct_model` and `gen_weights`.

---

## 4. Merge Steps

### Step 1 — Fix three issues in the existing mbirjax code

These three issues must be resolved before `MACE4DModel` can be merged cleanly.

**Shared class-level logger.**  All models of one class share a single logger object.
When `prox_map` runs concurrently across multiple threads, one thread can close a log
file handler while another thread is writing to it.  The fix is to give each model
instance its own logger, created in `__init__`.  This removes the need for
`MACE4DModel` to overwrite the logger on each `ConeBeamModel` after construction.

**Incorrect statistics in `auto_set_regularization_params`.**  Inside
`QGGMRFDenoiser`, this method calls `subsample_views` with
`num_real_views = sinogram_shape[0]`.  For a batch of hyperplanes, `sinogram_shape[0]`
equals `nt` (the number of time frames), not the batch size.  The result is that
regularization statistics are computed from the first hyperplane only, not from the
full batch.  The fix is to add a path in `QGGMRFDenoiser` that computes statistics
directly from the full array.

**Direct call to a private method.**  `MACE4DModel` currently calls
`QGGMRFDenoiser._denoise_single_device` directly, passing raw flat arrays.  Adding a
public `denoise_batch(x, sigma)` method to `QGGMRFDenoiser` eliminates this dependency
on a private method.

### Step 2 — Add `mbirjax/mace4d.py`

`MACE4DModel` should inherit from `ParameterHandler`, not `TomographyModel`, because it
orchestrates other models but does not perform projections itself.  The Category 2
parameters listed above are registered in `__init__` via `set_params`.

The following helper functions move into this file because they have no dependencies
outside `MACE4DModel`: `_dejitter_4d_dct`, `_normalize_prior_weights`, `_assign_tasks`,
`_resolve_devices`, `_denoiser_wrapper`, and `_batched_hyperplane_denoise`.

### Step 3 — Move `construct_time_frames` to `mbirjax/utilities.py`

`construct_time_frames` is a general preprocessing function with no dependency on
`MACE4DModel`.  It should be exported from `mbirjax/__init__.py` alongside
`copy_ct_model` and `gen_weights`.

### Step 4 — Reconcile the GIF utilities

`mbirjax/utilities.py` already contains `save_volume_as_gif`.  The only differences
from `gen_gif_and_save` in `mace4d.py` are a configurable colormap range and a
per-frame title.  These two features should be added to `save_volume_as_gif`, and
`gen_gif_and_save` should then be removed from `mace4d.py`.

### Step 5 — Update `mbirjax/__init__.py`

Export `MACE4DModel` and `construct_time_frames`.

### Step 6 — Port the tests

Move `tests/test_mace4d.py` to `mbirjax/tests/test_mace4d.py`.  Update the `recon()`
call to pass `sino_list` as the first positional argument.  The existing test coverage
is otherwise unchanged: it covers time-frame construction, dejitter correctness, task
assignment, batched denoising, end-to-end serial reconstruction, and the init image cache.

### Step 7 — Update docstrings and documentation

Convert all docstrings from NumPy style to Google style (`Args:`, `Returns:`, `Raises:`,
`Example:`), which is the convention used throughout mbirjax.  Update `README.md`,
`demo_4d.sh`, and `recon_4d.py` to reflect the new interface.

---

## 5. Open Questions

One question remains unresolved before the merge can be finalized.

**`weight_type` registration.**  This parameter is not currently registered in
mbirjax's parameter system.  It can be added to the registry, or kept as a plain
constructor keyword that is not accessible via `set_params`.
