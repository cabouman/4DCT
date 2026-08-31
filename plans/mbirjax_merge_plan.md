# MACE4DModel — mbirjax Merge Plan

## 1. Parameter Classification

mbirjax assigns parameters to one of four categories: constructor arguments,
optional parameters managed via `set_params`, required arguments to `recon()`,
and optional arguments to `recon()`.  The tables below place each parameter of
`MACE4DModel` into one of these categories.

### Category 1 — Constructor arguments (fixed after construction)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ct_model` | *(required)* | `mbirjax.ConeBeamModel`, fully built for the full scan.  Per-frame models and view slices are derived from its angles at construction time. |
| `frames_per_rotation` | `6` | Time frames per full rotation.  Sets the frame stride and the period of the temporal dejitter filter. |
| `frame_overlap_factor` | `2.0` | Number of frames that share any given view.  Each frame spans `frame_overlap_factor × (360 / frames_per_rotation)` degrees. |
| `num_frames` | `None` | Reconstruct only the first N time frames (smoke tests, partial runs).  `None` uses all frames. |

These are structural parameters, analogous to `angles` in `ConeBeamModel`: they
determine the frame decomposition and are fixed for the object's lifetime.  They
are **not** settable via `set_params`.  Changing them means constructing a new model.

The constructor derives everything from the model alone — no sinogram is needed:
`views_per_frame` and the stride come from the model's angle spacing, the view-slice
list follows, and the per-frame `ConeBeamModel`s are built via `copy_ct_model` on the
sliced angles.  The per-frame models and view slices are stored as attributes
(`self.model_list`, `self.view_slices`) and are inspectable immediately after
construction.

### Category 2 — Optional parameters, managed via `set_params`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mace_prior_weight` | `0.5` | Total weight of the three prior agents.  A scalar `w` maps to weights `[1−w, w/3, w/3, w/3]` across `[fwd, xyt, yzt, xzt]`.  A three-element list `[w0, w1, w2]` sets each prior weight independently. |
| `rho_mann` | `0.5` | Mann iteration step size (ADMM ρ). |
| `prox_num_iterations` | `3` | Maximum number of `prox_map` iterations per MACE step. |
| `prox_stop_threshold` | `0.02` | `prox_map` convergence threshold, expressed as percent change. |
| `sigma_prox` | `None` | Proximal sigma.  `None` lets mbirjax select a value automatically. |
| `dejitter` | `True` | Apply DCT-I temporal dejitter inside each agent. |
| `weight_type` | `"transmission_root"` | Weight type used by `gen_weights` when `recon()` is called with `weights=None`.  Ignored when an explicit weights array is passed.  One of `'unweighted'`, `'transmission'`, `'transmission_root'`, `'emission'`. |
| `verbose` | `1` | Verbosity level.  `0` = silent, `1` = progress, `2` = debug.  Already registered in mbirjax's parameter system. |

Note that iteration control for the outer MACE loop is **not** here: following the
`TomographyModel.recon` convention, `max_iterations` and `stop_threshold_change_pct`
are `recon()` arguments (Category 4).

### Category 3 — Required `recon()` arguments

| Parameter | Type | Notes |
|-----------|------|-------|
| `sinogram` | ndarray | Full sinogram, shape `(num_views, det_rows, det_cols)` — the same name and role as in `TomographyModel.recon`.  Sliced into per-frame sinograms internally using the stored view slices (NumPy views, no copies). |

### Category 4 — Optional `recon()` arguments

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `weights` | `None` | Full-sinogram-shaped weights array, sliced per frame internally (views).  `None` generates weights once on the full sinogram via `gen_weights(sinogram, weight_type)` and slices the result.  **Documented deviation from the base class**, where `weights=None` means unit weights; unit weights here are available via `set_params(weight_type='unweighted')`.  An explicit array supports custom weighting (defective detector columns, `gen_weights_mar`, etc.). |
| `init_recon` | `None` | A 4D array of shape `(nt, nx, ny, nz)` to use as the starting point (mbirjax name; was `init_image`).  When provided, the per-frame MBIR initialization is skipped. |
| `max_iterations` | `10` | Number of outer MACE iterations (was `max_mace_itr`). |
| `stop_threshold_change_pct` | `0` | Stop the outer loop when the per-iteration consensus change (already computed and logged as `consensus_change_pct`) falls below this percentage.  Default `0` runs exactly `max_iterations`, preserving the validated Lilly behavior; the base-class default of `0.2` was not adopted so that existing reference runs remain reproducible. |
| `init_dir` | `None` | Directory for caching the computed initialization image (`init_image.npy`).  On a subsequent run with the same settings, the cached image is loaded instead of recomputed. |
| `log_dir` | `None` | Directory for `run_info.txt`, `timing_log.csv`, and `task_log.csv`.  `None` writes no log files. |

**Return value.**  `recon()` returns `(recon_4d, recon_dict)`, matching
`TomographyModel.recon`.  `recon_dict` carries the run settings currently written to
`run_info.txt` plus the per-iteration consensus change and timing summary (the content
of `timing_log.csv`); the log files remain available via `log_dir` for long Slurm runs.

### Device configuration

Devices are configured via a `configure_devices(devices=None)` method, mirroring the
semantics of `TomographyModel.configure_devices` on current mbirjax `main` (the merge
target): `None` = all visible GPUs (the CPU when there are none), `int n` = the first
`n` devices, a sequence of jax devices = exactly those.  Never calling it is equivalent
to `configure_devices(None)`.  The `devices=` argument to `recon()` is removed; the
`_resolve_devices` helper is absorbed into `configure_devices`.  One device runs all
tasks inline (the serial path), as before.

---

## 2. Interface Decision — Operator in the Constructor, Data at `recon()`

This reverses the earlier Option A decision (sinogram in the constructor).  The key
observation is that **the entire frame structure is derivable from the model alone**:
`construct_time_frames` uses the sinogram only for `num_views` (equal to `len(angles)`,
already in the model) and to slice it.  The per-frame models, `views_per_frame`, the
stride, and the view-slice list all come from `angles` + `frames_per_rotation` +
`frame_overlap_factor`.  Slicing the sinogram is a set of NumPy views — essentially
free — and weight generation is a cheap elementwise operation.  So the structural half
of the old `construct_time_frames` belongs in the constructor, and the data half
belongs in `recon()`.

This gives both design goals at once:

* **mbirjax look and feel.**  Constructor defines an operator from geometry;
  `recon(sinogram, weights=None, init_recon=None, max_iterations=..., ...)` takes the
  data and returns `(recon, recon_dict)` — exactly the contract of every other model.
* **Simple 4D workflow.**  The user never handles parallel `sino_list`/`model_list`;
  the workflow is the same length as under Option A.
* **Weights are visible and controllable again.**  Custom weights arrays are supported;
  the `weight_type` registration question resolves itself (Section 4).
* **Model reuse.**  Many scans of one protocol can be reconstructed with a single
  model instance (`for s in scans: mace.recon(s)`), keeping the per-frame models'
  compiled projectors and the thread-local denoiser caches warm — relevant for
  production pipelines.
* **Memory lifetime.**  Weights are generated once on the full sinogram inside
  `recon()` and sliced as views, instead of materializing ~`frame_overlap_factor` ×
  the full sinogram in host RAM for the object's lifetime (and instead of computing
  overlapping views' weights twice).

### Workflow example

```python
import mbirjax as mj
import mbirjax.preprocess as mjp

# Preprocessing (unchanged from 3D workflows): data + geometry.
sinogram, ct_model = mjp.nsi.get_sino_and_model(dataset_dir, auto_crop=True)
ct_model.set_params(sharpness=1.0, positivity_flag=True)

# Constructor: frame structure only — no data.
mace = mj.MACE4DModel(ct_model, frames_per_rotation=6, frame_overlap_factor=2.0)
mace.set_params(mace_prior_weight=0.5, rho_mann=0.5)
mace.configure_devices()          # optional; default uses all visible GPUs

# Reconstruction: data enters here, like every other mbirjax model.
recon_4d, recon_dict = mace.recon(sinogram, max_iterations=10,
                                  init_dir="./output/init", log_dir="./output/logs")

# Same protocol, another scan: reuse the model, caches stay warm.
recon_4d_b, _ = mace.recon(sinogram_b)
```

Shape errors still surface early: `recon()` validates `sinogram.shape` against the
model's `sinogram_shape`, as the base class does.  For pre-recon inspection,
`mace.model_list` and `mace.view_slices` are available right after construction, and
per-frame sinograms are `[sinogram[sl] for sl in mace.view_slices]`.

---

## 3. Merge Steps

### Step 1 — Fix one issue in the existing mbirjax code

One issue in the existing mbirjax code must be resolved before `MACE4DModel` can be
merged cleanly (still present on current `main`).

**Shared class-level logger.**  All models of one class share a single logger object.
When `prox_map` runs concurrently across multiple threads, one thread can close a log
file handler while another thread is writing to it.  The fix is to give each model
instance its own logger, created in `__init__`.  Pitfall: `logging.getLogger(name)`
caches by name, and `setup_logger` currently uses `self.__class__.__name__` — a
per-instance logger therefore needs a unique name (e.g. include `id(self)`) or must
bypass the logging registry; otherwise all instances still share one object.  This
removes the need for `MACE4DModel`'s `_silence_model_logging` workaround, which is
deleted.

The other two issues identified earlier (incorrect statistics in
`auto_set_regularization_params` and direct calls to `_denoise_single_device`) do not
require changes to mbirjax.  Both are handled within `MACE4DModel`'s own code:
`_configure_denoiser` calls the individual `auto_set_*` methods directly on the full
array, and `_batched_hyperplane_denoise` calls `_denoise_single_device` as internal
code within the same package.

### Step 2 — Add `mbirjax/mace4d.py`

`MACE4DModel` should inherit from `ParameterHandler`, not `TomographyModel`, because it
orchestrates other models but does not perform projections itself.  The Category 2
parameters listed above are registered in `__init__` via `set_params(no_warning=True, ...)`,
with a `MACE4DParamNames` Literal extension following the `ConeBeamParamNames` pattern.

Constructor: takes `ct_model`, `frames_per_rotation`, `frame_overlap_factor`,
`num_frames`; derives the view slices from the model's angles and builds the per-frame
models via `copy_ct_model` — no sinogram, no weights.

`recon()`: takes `sinogram` (validated against the model's `sinogram_shape`) and the
Category 4 arguments; slices sinogram and weights per frame as views; generates weights
once on the full sinogram when `weights=None`; implements the outer
`stop_threshold_change_pct` using the already-computed consensus change; assembles and
returns `(recon_4d, recon_dict)`.

Add `configure_devices(devices=None)` mirroring the `TomographyModel` semantics;
remove `devices=` from `recon()` and absorb `_resolve_devices`.

The following helper functions move into this file because they have no dependencies
outside `MACE4DModel`: `_dejitter_4d_dct`, `_normalize_prior_weights`, `_assign_tasks`,
`_denoiser_wrapper`, `_batched_hyperplane_denoise`, `_configure_denoiser`,
`_denoise_constants`, `_get_qggmrf_denoiser`, and `_auto_batch_size`.
(`_resolve_devices` is absorbed into `configure_devices`; `_silence_model_logging` is
deleted per Step 1.)

Cleanups to make while moving the code:

* Fix the stale `_configure_denoiser` docstring, which still says the `auto_set_*`
  calls "become overrides ... inside QGGMRFDenoiser at merge time" — that contradicts
  the settled decision to leave `QGGMRFDenoiser` untouched.
* The code monkey-patches `denoiser._log_denoise_progress` and stashes `_mace4d_*`
  attributes on denoiser instances.  Acceptable intra-package once merged, but fragile
  against future `denoising.py` refactors; a small upstream alternative is making the
  progress callback vmap-safe.  Not required for the merge.
* Verify that registering `sigma_prox` behaves sensibly: user calls to
  `set_params(sigma_prox=...)` trip `ParameterHandler`'s "disables auto-regularization"
  warning path, which is probably appropriate here but should be checked for an
  orchestrator that has no regularization of its own.

### Step 3 — Add `construct_time_frame_models` to `mbirjax/utilities.py`

The public frame utility becomes model-only:
`construct_time_frame_models(model, frames_per_rotation=6, frame_overlap_factor=2.0)`
returns `(model_list, view_slices)`.  It replaces the old sinogram-splitting
`construct_time_frames`; per-frame sinograms are obtained by slicing with the returned
view slices.  The `MACE4DModel` constructor calls it, and it is exported from
`mbirjax/__init__.py` alongside `copy_ct_model` and `gen_weights` for users who want to
inspect the frame decomposition directly.

### Step 4 — Reconcile the GIF utilities

`mbirjax/utilities.py` already contains `save_volume_as_gif`.  The only differences
from `gen_gif_and_save` in `mace4d.py` are a configurable colormap range and a
per-frame title.  These two features should be added to `save_volume_as_gif`, and
`gen_gif_and_save` should then be removed from `mace4d.py`.

### Step 5 — Update `mbirjax/__init__.py`

Export `MACE4DModel` and `construct_time_frame_models`.

### Step 6 — Port the tests

Move `tests/test_mace4d.py` to `mbirjax/tests/test_mace4d.py`.  Update to the new
interface: construct with `ct_model` (+ frame parameters) instead of pre-split lists,
pass `sinogram` to `recon()`, unpack `(recon, recon_dict)`, rename `init_image` →
`init_recon`, and use `configure_devices` where the tests select devices.  Existing
coverage (time-frame construction, dejitter correctness, task assignment, batched
denoising, end-to-end serial reconstruction, init image cache) carries over; add
coverage for the new behavior: explicit `weights` vs `weights=None` generation,
the outer `stop_threshold_change_pct`, `num_frames` truncation, and
`configure_devices` resolution (None / int / list).

### Step 7 — Update docstrings and documentation ✓ (Google-style conversion done)

All docstrings in `mace4d.py` have been converted from NumPy style to Google style
(`Args:`, `Returns:`, `Raises:`, `Example:`), matching the convention used throughout
mbirjax.  The interface changes above require a content pass at merge time (renamed and
moved arguments, return tuple, `configure_devices`).  Remaining work: update
`README.md`, `demo_4d.sh`, and `recon_4d.py` to the new interface.

---

## 4. Resolved Questions

**`weight_type` registration** — resolved by the interface reversal.  `weight_type`
stays a registered Category 2 parameter, but it now only governs the `weights=None`
case of `recon()`; explicit weights arrays bypass it entirely.  The one deliberate
deviation from the base class — `weights=None` meaning auto-generated
`transmission_root` weights rather than unit weights — is a documentation decision,
stated prominently in the `recon()` docstring, with unit weights available via
`weight_type='unweighted'`.
