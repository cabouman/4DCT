# MACE4DModel → mbirjax Merge Plan

## 1. Parameter Classification

Following mbirjax convention: required args go to the constructor; optional args are
registered in the parameter system and settable via `set_params`; `recon()` receives
only the sinogram data and call-time options.

### Category 1 — Required constructor arguments (no defaults)

| Parameter | Type | Notes |
|-----------|------|-------|
| `model_list` | `list[ConeBeamModel]` | One per time frame; defines geometry. Analogous to `angles` in ConeBeamModel. |

> **Design decision:** `sino_list` moves to `recon()` (Category 3) to match mbirjax's
> pattern where sinogram *data* is passed at call time, not construction time.
> Weights are computed lazily at the start of `recon()`.

### Category 2 — Optional constructor / `set_params` parameters

These are registered in the parameter system so callers can use `set_params` after
construction, the same way mbirjax users adjust `sharpness` or `sigma_y`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `mace_prior_weight` | `0.5` | Total weight of the three prior agents; `[1−w, w/3, w/3, w/3]` across `[fwd, xyt, yzt, xzt]`. Accepts scalar or `[w0, w1, w2]`. |
| `rho_mann` | `0.5` | Mann iteration step size (ADMM ρ). |
| `max_mace_itr` | `10` | Number of outer MACE iterations. |
| `prox_num_iterations` | `3` | Max `prox_map` iterations per MACE step. Maps to existing `max_iterations` convention. |
| `prox_stop_threshold` | `0.02` | `prox_map` convergence threshold (% change). |
| `sigma_p` | `None` | Proximal sigma; `None` lets mbirjax choose. Maps naturally to `sigma_prox`. |
| `dejitter` | `True` | Apply DCT-I temporal dejitter inside each agent. |
| `frames_per_rotation` | `6` | Jitter period (frames per full rotation); also used by `construct_time_frames`. |
| `weight_type` | `"transmission_root"` | Sinogram weight type for `gen_weights`. |
| `verbose` | `1` | Already in `ParamNames`; 0 = silent, 1 = progress, 2 = debug. |

### Category 3 — Required `recon()` arguments

| Parameter | Type | Notes |
|-----------|------|-------|
| `sino_list` | `list[ndarray]` | Per-frame sinograms, each `(views_per_frame, det_rows, det_cols)`. |

### Category 4 — Optional `recon()` arguments

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `devices` | `None` | `None` = all visible GPUs; `int n` = first n devices; `list` = explicit. |
| `init_image` | `None` | Provide a 4D `(nt, nx, ny, nz)` starting point; skips per-frame MBIR init. |
| `init_dir` | `None` | Cache directory for the computed init image (`init_image.npy`). |
| `log_dir` | `None` | Directory for `run_info.txt`, `timing_log.csv`, `task_log.csv`. `None` = no logs. |

---

## 2. Proposed Interface

```python
# Construction (geometry only)
mace = mj.MACE4DModel(model_list, mace_prior_weight=0.5, max_mace_itr=10,
                      frames_per_rotation=6)
mace.set_params(rho_mann=0.5, dejitter=True)

# Reconstruction (data passed here, matching mbirjax's recon() convention)
recon_4d = mace.recon(sino_list, init_dir="./output/init", log_dir="./output/logs")
```

`construct_time_frames` stays a standalone helper (not a method), consistent with how
mbirjax exposes preprocessing utilities at module level.

---

## 3. Merge Steps

**Step 1 — Fix prerequisites inside mbirjax** *(touches existing files)*

1. **Per-instance loggers in `TomographyModel`:** The current shared class-level logger
   races when `prox_map` runs concurrently in threads. Replace with one logger per
   instance (created in `__init__`, not lazily in `setup_logger`). This removes the
   monkey-patching in `MACE4DModel._silence_model_logging`.
2. **`auto_set_regularization_params` for a hyperplane batch:** The current
   implementation calls `subsample_views(..., num_real_views=sinogram_shape[0])`, which
   treats `nt` (not batch size) as the view count and silently uses only the first
   hyperplane's statistics. Expose a `_auto_set_regularization_params_from_array(image)`
   override path in `QGGMRFDenoiser` that operates on the whole array directly.
3. **Public batched-denoise interface in `QGGMRFDenoiser`:** Add a
   `denoise_batch(x, sigma)` method that wraps the vmapped `_denoise_single_device`
   path so `MACE4DModel` no longer calls a private method with raw flat arrays.

**Step 2 — Add `mbirjax/mace4d.py`** *(new file)*

- `MACE4DModel` inherits `ParameterHandler` (not `TomographyModel`; it orchestrates
  models but does no projections itself).
- Register all Category 2 parameters in `__init__` via `set_params`.
- Move `_dejitter_4d_dct`, `_normalize_prior_weights`, `_assign_tasks`,
  `_resolve_devices`, `_denoiser_wrapper`, and `_batched_hyperplane_denoise` here (they
  have no mbirjax dependencies and belong with the class they serve).

**Step 3 — Move `construct_time_frames` to `mbirjax/utilities.py`**

It is a general preprocessing helper with no dependency on `MACE4DModel`. Export it
from `__init__.py` alongside `copy_ct_model` and `gen_weights`.

**Step 4 — Reconcile `gen_gif_and_save` with `save_volume_as_gif`**

`mbirjax/utilities.py` already has `save_volume_as_gif`. Extend it with `vmin`/`vmax`
and per-frame title support (the only differences), then delete `gen_gif_and_save` from
`mace4d.py`.

**Step 5 — Update `mbirjax/__init__.py`**

Export `MACE4DModel` and `construct_time_frames`.

**Step 6 — Port tests**

Move `tests/test_mace4d.py` to `mbirjax/tests/test_mace4d.py`. Update the `recon()`
call to pass `sino_list` as a positional argument (per the new interface). Existing test
coverage (time-frame construction, dejitter, task assignment, batched denoiser,
end-to-end serial recon, init cache) is retained as-is.

**Step 7 — Update docs and docstrings**

Convert all docstrings from NumPy style (`Parameters\n----------`) to mbirjax's Google
style (`Args:`, `Returns:`, `Raises:`, `Example:`). Update `README.md`, `demo_4d.sh`,
and `recon_4d.py` to use the new `mj.MACE4DModel` interface.

---

## 4. Open Questions

- **`sigma_p` vs `sigma_prox`:** mbirjax already has `sigma_prox` in `ParamNames`.
  Alias `sigma_p → sigma_prox` at the `set_params` level, or rename the CLI flag.
- **`weight_type` in `ParamNames`:** Not currently a registered param. Either add it or
  keep it as a plain constructor kwarg that is not set_params-accessible.
- **`frames_per_rotation` scope:** It serves double duty (dejitter period *and* argument
  to `construct_time_frames`). Keeping it on the model is correct since it defines the
  jitter period intrinsic to the acquisition protocol.
