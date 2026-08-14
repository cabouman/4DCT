# 4DCT MACE Recon

4D CT reconstruction using MACE with a cone-beam prox_map forward agent and three qGGMRF prior agents across XY-t, YZ-t, and XZ-t hyperplanes. Designed for the Eli Lilly 4DCT phantom dataset acquired on NSI hardware.

## Quick Start (Lilly)

Edit `DATA_PATH` in `run_lilly.sh` to point to your extracted NSI dataset, then:

```bash
bash test_script_4D.sh
```

Output is written to `./output/`:
- `recon_4d_<time>h.npy` — reconstructed 4D volume, shape `(nt, nx, ny, nz)`
- `init/init_image.npy` — per-bin MBIR initialization (saved for potential reuse)
- `timing_log.csv` — per-iteration agent timing

To skip re-initialization on a second run, add `--resume` to the python command in `run_lilly.sh`.

## Layout

```
4DCT/
├── run_lilly.sh          # shell entry point — edit DATA_PATH and run
├── lilly_recon.py        # production CLI script (few argparse flags)
├── dev_recon.py          # dev/exploration script (all params as Python variables)
├── model_4d.py           # MACE4DModel class (serial + multi-GPU)
├── utils.py              # shared utilities (time binning, dejitter, denoiser helpers)
├── plans/
│   └── refactor_plan.md  # design decisions and interface discussion
├── 4DMACE_serial/        # original serial implementation (reference)
└── 4DMACE_multi_threads/ # original multi-GPU implementation (reference)
```

## File Roles

| File | Purpose |
|------|---------|
| `run_lilly.sh` | Shell entry point. Edit `DATA_PATH`; everything else has sensible defaults. |
| `lilly_recon.py` | Production script. Minimal CLI: data path, downsampling, bin params, iteration count. Algorithmic hyperparameters are fixed. |
| `dev_recon.py` | Dev script. All parameters are Python variables at the top. Supports `USE_SAVED_INIT_IMAGE`, `parallel`, `time_range`, custom `device_indices`, etc. |
| `model_4d.py` | `MACE4DModel` class. Call `model.recon(parallel=True)` for multi-GPU or `model.recon(parallel=False)` for serial. |
| `utils.py` | Stateless helpers shared by both modes: `truncate_sino_into_time_bins`, `dejitter_4d_dct`, denoiser utilities. |

## MACE4DModel Interface

```python
from model_4d import MACE4DModel
from utils import truncate_sino_into_time_bins

# 1. Preprocess with mbirjax
sino, ct_model = mjp.nsi.get_sino_and_model(dataset_dir, ...)
ct_model.set_params(sharpness=1.0, positivity_flag=True)

# 2. Split into time bins
bins = truncate_sino_into_time_bins(sino, ct_model, views_per_bin=48, stride=24)
sino_list = [b[0] for b in bins]
model_list = [b[1] for b in bins]

# 3. Build model and reconstruct
model_4d = MACE4DModel(sino_list, model_list, prior_weight=0.5, max_mace_itr=10)
recon_4d = model_4d.recon(parallel=True, init_save_dir="./output/init")
```

## Multi-GPU Architecture

Four agents run concurrently via `ThreadPoolExecutor(4)`:

| Agent | Role | GPU |
|-------|------|-----|
| 0 | Cone-beam `prox_map` (forward model) | GPU 0 |
| 1 | qGGMRF denoiser, XY-t hyperplanes | GPU 1 |
| 2 | qGGMRF denoiser, YZ-t hyperplanes | GPU 2 |
| 3 | qGGMRF denoiser, XZ-t hyperplanes | GPU 3 |

**GPU pinning**: every `ConeBeamModel` and `QGGMRFDenoiser` is pinned to exactly one GPU via `configure_devices([device])`. This prevents the 4-way NCCL clique deadlock that occurs when mbirjax's auto-sharding builds a multi-GPU Mesh inside concurrent threads.

Requires ≥4 JAX-visible GPUs. Use `parallel=False` for single-GPU or CPU-only runs.

## DCT-I Temporal Dejitter

`dejitter_4d_dct` removes periodic gating jitter (period=6 frames) by zeroing the corresponding DCT-I frequency bands. It is applied:
- After the forward agent output
- Before each prior agent input

This is controlled by `dejitter=True` (default) and `dejitter_period=6` on `MACE4DModel`.

## Timing Log

`model.recon(parallel=True, timing_log_path="./output/timing_log.csv")` writes a CSV with columns:

```
iteration, agent_0_forward_sec, agent_1_prior_xyt_sec,
agent_2_prior_yzt_sec, agent_3_prior_xzt_sec, iteration_total_sec
```
