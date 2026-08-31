# MACE4DModel — mbirjax Merge Plan

## 1. Parameter Classification

mbirjax assigns parameters to one of four categories: required constructor arguments,
optional constructor arguments managed via `set_params`, required arguments to `recon()`,
and optional arguments to `recon()`.  The tables below place each parameter of
`MACE4DModel` into one of these categories.

### Category 1 — Required constructor arguments

| Parameter | Type | Notes |
|-----------|------|-------|
| `sino` | ndarray | Full sinogram, shape `(num_views, det_rows, det_cols)`. Frame splitting runs at construction time via `construct_time_frames`. |
| `model` | `mbirjax.ConeBeamModel` | Fully-built model for the full scan. Per-frame models are derived at construction time. |

The constructor calls `construct_time_frames(sino, model, frames_per_rotation,
frame_overlap_factor)` internally to produce the per-frame sinograms, per-frame models,
and sinogram weights.  These are stored as attributes (`self.sino_list`,
`self.model_list`, `self.weights_list`) and are ready when construction finishes.
`construct_time_frames` remains a public module-level function for users who want to
inspect the per-frame data before reconstruction.

### Category 2 — Optional constructor arguments, managed via `set_params`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `frames_per_rotation` | `6` | Number of time frames per full rotation.  Sets the stride for frame splitting and the period of the temporal dejitter filter. |
| `frame_overlap_factor` | `2.0` | Number of frames that share any given view.  Each frame spans `frame_overlap_factor × (360 / frames_per_rotation)` degrees. |
| `mace_prior_weight` | `0.5` | Total weight of the three prior agents.  A scalar `w` maps to weights `[1−w, w/3, w/3, w/3]` across `[fwd, xyt, yzt, xzt]`.  A three-element list `[w0, w1, w2]` sets each prior weight independently. |
| `rho_mann` | `0.5` | Mann iteration step size (ADMM ρ). |
| `max_mace_itr` | `10` | Number of outer MACE iterations. |
| `prox_num_iterations` | `3` | Maximum number of `prox_map` iterations per MACE step. |
| `prox_stop_threshold` | `0.02` | `prox_map` convergence threshold, expressed as percent change. |
| `sigma_prox` | `None` | Proximal sigma.  `None` lets mbirjax select a value automatically. |
| `dejitter` | `True` | Apply DCT-I temporal dejitter inside each agent. |
| `weight_type` | `"transmission_root"` | Sinogram weight type passed to `gen_weights`. |
| `verbose` | `1` | Verbosity level.  `0` = silent, `1` = progress, `2` = debug.  Already registered in mbirjax's parameter system. |

### Category 3 — Required `recon()` arguments

None.  Under this interface the sinogram enters at construction time, so `recon()` takes
no required data arguments.

### Category 4 — Optional `recon()` arguments

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `devices` | `None` | `None` uses all visible GPUs.  An integer `n` uses the first `n` devices.  A list specifies devices explicitly. |
| `init_image` | `None` | A 4D array of shape `(nt, nx, ny, nz)` to use as the starting point.  When provided, the per-frame MBIR initialization is skipped. |
| `init_dir` | `None` | Directory for caching the computed initialization image (`init_image.npy`).  On a subsequent run with the same settings, the cached image is loaded instead of recomputed. |
| `log_dir` | `None` | Directory for `run_info.txt`, `timing_log.csv`, and `task_log.csv`.  `None` writes no log files. |

---

## 2. Interface Decision — Sinogram Enters at Construction Time

We considered two options: passing `sino` to the constructor (Option A) or passing it
to `recon()` (Option B, which matches the ConeBeamModel convention).  We chose Option A.

The reason is that `MACE4DModel` is always constructed immediately after the full CT
model is built, with the sinogram already in hand.  Unlike `ConeBeamModel`, which can be
reused with different sinograms, `MACE4DModel` is tied to one specific 4D acquisition.
Frame splitting and weight computation are initialization work, not reconstruction work —
they determine the model's structure and are fixed for the object's lifetime.  Doing them
at construction time gives the object a fully defined state from the moment it is created.

### Proposed interface

```python
# Construction: frame splitting and weight computation happen here
mace = mj.MACE4DModel(
    sino=sino,
    model=ct_model,
    frames_per_rotation=6,
    frame_overlap_factor=2.0,
    mace_prior_weight=0.5,
    max_mace_itr=10,
)
mace.set_params(rho_mann=0.5, dejitter=True)

# Reconstruction: no sinogram argument needed
recon_4d = mace.recon(init_dir="./output/init", log_dir="./output/logs")
```

`construct_time_frames` remains a public module-level function for users who want to
inspect the per-frame sinograms and models before calling `recon()`.

---

## 3. Merge Steps

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

The constructor takes `sino` and `model`, calls `construct_time_frames` internally, and
stores the resulting per-frame sinograms, models, and weights as attributes.

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

Move `tests/test_mace4d.py` to `mbirjax/tests/test_mace4d.py`.  Update the
`MACE4DModel` constructor call to pass `sino` and `model` directly instead of
pre-split `sino_list` and `model_list`.  Update `recon()` calls to remove the
`sino_list` argument.  The existing test coverage is otherwise unchanged: it covers
time-frame construction, dejitter correctness, task assignment, batched denoising,
end-to-end serial reconstruction, and the init image cache.

### Step 7 — Update docstrings and documentation ✓ (docstrings done)

All docstrings in `mace4d.py` have been converted from NumPy style to Google style
(`Args:`, `Returns:`, `Raises:`, `Example:`), matching the convention used throughout
mbirjax.  Remaining work at merge time: update `README.md`, `demo_4d.sh`, and
`recon_4d.py` to reflect the new interface.

---

## 4. Open Questions

One question remains unresolved before the merge can be finalized.

**`weight_type` registration.**  This parameter is not currently registered in
mbirjax's parameter system.  It can be added to the registry, or kept as a plain
constructor keyword that is not accessible via `set_params`.
