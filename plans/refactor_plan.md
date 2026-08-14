# 4DCT Refactor Plan

## Goal

Restructure the 4DCT repo to mirror the `mbirjax_applications/nsi` layout: a production-ready Lilly script driven by a shell file, a developer exploration script with full parameter access, and a clean class-based 4D model sitting underneath both.

---

## New File Structure

```
4DCT/
├── plans/
│   └── refactor_plan.md          ← this file
├── utils.py                      ← shared utilities (dejitter, binning, denoiser helpers)
├── model_4d.py                   ← MACE4DModel class (serial + parallel)
├── Lilly_recon.py                ← production CLI script (few argparse params)
├── dev_recon.py                  ← dev/exploration script (full parameter control)
├── test_script_4D.sh                  ← shell script: edit DATA_PATH and run
├── output/                       ← gitignored
├── data/                         ← gitignored
├── 4DMACE_serial/                ← kept unchanged for reference
└── 4DMACE_multi_threads/         ← kept unchanged for reference
```

---

## File Responsibilities

### `utils.py`
All shared, stateless utilities. No main block. Merges the best of `utils_serial.py` and `utils_multi_threads.py`.

Functions:
- `truncate_sino_into_time_bins(sino, model, views_per_bin, stride)` — splits full sinogram into overlapping time bins (identical in both existing files)
- `dejitter_4d_dct(recon_4d, period, ...)` — canonical DCT-I jitter removal (from `utils_multi_threads.py`)
- `normalize_prior_weights(prior_weight)` — float → [forward_w, xyt_w, yzt_w, xzt_w]
- `get_qggmrf_denoiser(shape, device)` — thread-local, device-pinned denoiser cache (critical for multi-GPU correctness)
- `estimate_sigma_per_hyperplane(x, device)` — per-hyperplane noise std on one device
- `qggmrf_hyperplane_denoise(x, sigma_list, device)` — batch denoising on one device
- `denoiser_wrapper(x, permute_vector, sigma_list, device)` — permute → denoise → unpermute

### `model_4d.py`
The `MACE4DModel` class. Takes preprocessed data (sino_list, model_list) and exposes a single `recon()` method.

```python
class MACE4DModel:
    def __init__(self, sino_list, model_list,
                 prior_weight=0.5, rho=0.5, max_mace_itr=10,
                 forward_num_iterations=3, stop_threshold=0.02,
                 weight_type="transmission_root", sigma_p=None, verbose=1,
                 dejitter=True, dejitter_period=6):
        ...

    def recon(self, init_image=None, parallel=True,
              init_save_dir=None, timing_log_path=None, device_indices=None):
        ...  # dispatches to _recon_parallel or _recon_serial

    def _recon_parallel(self, ...):
        ...  # ThreadPoolExecutor(4) with GPU pinning — core multi-GPU logic

    def _recon_serial(self, ...):
        ...  # sequential agents, single device
```

Key design decisions:
- Weights are computed once in `__init__` (not passed in separately)
- `parallel=True` is the default; `_recon_parallel` requires ≥4 GPUs
- `dejitter` and `dejitter_period` are class-level; the DCT is applied inside agents
- `init_image=None` triggers automatic per-bin MBIR initialization and saves to `init_save_dir`
- All threading and GPU-pinning logic is preserved verbatim inside `_recon_parallel`

### `Lilly_recon.py`
Production script. Minimal argparse interface. All algorithmic hyperparameters (prior_weight, rho, etc.) are hard-coded at the top. Always runs parallel.

Exposed CLI flags:
```
--data_path            Path to NSI dataset directory (required)
--downsample           Detector subsampling factor (default: 1)
--subsample_view_factor View subsampling factor (default: 1)
--max_mace_itr         MACE outer iterations (default: 10)
--output_path          Output directory (default: ./output)
--resume               Load saved init_image from output/init/ if it exists
```

Init-image policy: by default always starts fresh (Lilly typically runs each dataset once). Pass `--resume` to reuse a saved init from a previous run.

Output: saves `recon_4d_<time>h.npy` and `timing_log.csv` to `--output_path`.

### `dev_recon.py`
Development/exploration script. All parameters are Python variables at the top of the file — no argparse. Intended to be edited directly before each experimental run.

Key variables:
```python
USE_SAVED_INIT_IMAGE = False   # flip to True to skip slow init
dataset_url = "..."            # path to .tgz or extracted dir
downsample_rate = [1, 1]
views_per_bin = 48
stride = 24
time_range = slice(0, -1)     # subset of bins for quick tests
parallel = True                # False → serial mode
device_indices = [0, 1, 2, 3]
# ... all MACE hyperparameters
```

### `test_script_4D.sh`
Shell script that Lilly operators edit once (set `DATA_PATH`) and then just run. Mirrors `nsi/test_script_mar.sh`.

```bash
#!/bin/bash
DATA_PATH=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/

python Lilly_recon.py \
  --data_path "$DATA_PATH" \
  --views_per_bin 48 \
  --stride 24 \
  --max_mace_itr 10 \
  2>&1 | tee ~/4dct_logs/lilly_run.log
```

---

## Multi-GPU Parallelism — Preserved Invariants

The following must not change across the refactor:

1. **ThreadPoolExecutor(4)**: Four agents run concurrently, one per GPU.
2. **SINGLE-GPU FIX**: Each `ConeBeamModel` and `QGGMRFDenoiser` is pinned to exactly one GPU via `configure_devices([device])`. This prevents the 4-way NCCL clique deadlock.
3. **Thread-local denoiser cache**: `get_qggmrf_denoiser` uses `threading.local()` keyed by `(shape, device)` so no denoiser instance is shared across threads.
4. **LD_LIBRARY_PATH re-exec**: Must appear at the very top of entry-point scripts (before any JAX import) to strip incompatible system CUDA libs.
5. **`W_snap` snapshot**: Each agent sees a consistent snapshot of `W` from the start of the iteration (not the live updated values).
6. **Dejitter inside agents**: `dejitter_4d_dct` is called both in the forward agent (on `X[0]` output) and in each prior agent (on `W[k]` input) so temporal alignment is enforced consistently.

---

## Interface Discussion — What to Expose vs. Hard-code

### Always Lilly-accessible (CLI flags)
| Parameter | Rationale |
|-----------|-----------|
| `--data_path` | Required: dataset location |
| `--downsample` | Adjusts memory/speed tradeoff |
| `--subsample_view_factor` | Same |
| `--max_mace_itr` | Controls reconstruction quality vs. time |
| `--output_path` | Needed for cluster runs |
| `--resume` | One-time re-run without re-init |

### Hard-coded in `Lilly_recon.py` (not exposed)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `prior_weight` | 0.5 | Validated default; tuning not expected |
| `rho` | 0.5 | ADMM step size, stable at 0.5 |
| `forward_num_iterations` | 3 | Sufficient for prox_map convergence |
| `stop_threshold` | 0.02 | Standard stopping criterion |
| `sharpness` | 1.0 | mbirjax default, works well |
| `sigma_p` | None | Auto-selected by mbirjax |
| `dejitter` | True | Always needed for 4DCT phantom data |
| `dejitter_period` | 6 | Fixed by the 6-phase gating protocol |
| `weight_type` | "transmission_root" | Standard for CT |
| `device_indices` | [0,1,2,3] | Fixed cluster topology |

### Developer-only (in `dev_recon.py` as Python variables)
Everything above, plus: `time_range`, `parallel`, `device_indices`, `sigma_p`, `USE_SAVED_INIT_IMAGE`.

---

## Serial vs. Parallel

Both modes are kept via the `parallel` flag on `MACE4DModel.recon()`:
- `parallel=True` (default): 4-GPU ThreadPoolExecutor. Requires ≥4 GPUs.
- `parallel=False`: Sequential agents on a single device (`jax.devices()[0]`).

The old `4DMACE_serial/` and `4DMACE_multi_threads/` directories are kept untouched as reference implementations. Over time, if serial mode is no longer needed, `_recon_serial` can be removed.

---

## Migration Steps

1. Create `plans/refactor_plan.md` (this file)
2. Create `utils.py` — merge + deduplicate both `utils_*.py` files
3. Create `model_4d.py` — `MACE4DModel` class wrapping both recon modes
4. Create `Lilly_recon.py` — production CLI script
5. Create `dev_recon.py` — dev exploration script
6. Create `test_script_4D.sh` — shell entry point
7. Update `README.md` — describe new structure
