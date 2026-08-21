# 4DCT MACE Recon

4D CT reconstruction using MACE with a cone-beam prox_map forward agent and three qGGMRF prior agents across XY-t, YZ-t, and XZ-t hyperplanes. Designed for the Eli Lilly 4DCT phantom dataset acquired on NSI hardware.

## Dependencies

Requires `mbirjax` plus `scipy`, `imageio`, and `matplotlib`.

## Quick Start (Lilly)

Edit `DATA_PATH` in `demo_4d.sh` to point to your extracted NSI dataset, then:

```bash
bash demo_4d.sh
```

Output is written to `./output/`:
- `recon_4d_<time>h.npy` — reconstructed 4D volume, shape `(nt, nx, ny, nz)`
- `init/init_image.npy` — per-frame MBIR initialization (saved for potential reuse)
- `logs/run_info.txt` — human-readable summary of all run settings
- `logs/timing_log.csv` — per-iteration agent timing and consensus change

On a second run with the same settings, the saved initialization in `output/init/` is detected and reused automatically; if its shape does not match the run, it is recomputed with a warning.

## Layout

```
4DCT/
├── demo_4d.sh            # shell entry point — edit DATA_PATH and run
├── recon_4d.py           # command-line reconstruction driver (all params as flags)
├── mace4d.py             # MACE4DModel class + all 4D helpers (frames, dejitter, denoisers, GIF)
├── plans/
│   ├── refactor_plan.md  # design decisions and interface discussion
│   └── lilly_interface.md # parameter reference for Lilly operators
└── tests/                # fast CPU tests (python -m pytest tests/)
```

The original serial and multi-GPU implementations (`4DMACE_serial/`, `4DMACE_multi_threads/`) were merged into `mace4d.py`; git history preserves them.

## File Roles

| File | Purpose |
|------|---------|
| `demo_4d.sh` | Shell entry point. Edit `DATA_PATH`; everything else has sensible defaults. |
| `recon_4d.py` | Reconstruction driver. Every parameter is a CLI flag with a validated default: data path, downsampling, frame geometry, MACE hyperparameters, execution mode (`--serial`, `--device_indices`). |
| `mace4d.py` | The complete 4D functional block: `MACE4DModel` (call `model.recon(parallel=True)` for multi-GPU, `parallel=False` for serial) plus time-frame construction, DCT-I dejitter, hyperplane-denoiser helpers, and the GIF writer. |

## MACE4DModel Interface

```python
from mace4d import MACE4DModel, construct_time_frames

# 1. Preprocess with mbirjax
sino, ct_model = mjp.nsi.get_sino_and_model(dataset_dir, ...)
ct_model.set_params(sharpness=1.0, positivity_flag=True)

# 2. Construct time frames
sino_frames, model_frames = construct_time_frames(sino, ct_model,
                                                  angle_span_per_frame=np.radians(120.0),
                                                  angle_stride=np.radians(60.0))

# 3. Build model and reconstruct
mace_model = MACE4DModel(sino_frames, model_frames, prior_weight=0.5, max_mace_itr=10)
recon_4d = mace_model.recon(parallel=True, init_dir="./output/init")
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

The dejitter step removes periodic gating jitter (period=6 frames) by zeroing the corresponding DCT-I frequency bands. It is applied:
- After the forward agent output
- Before each prior agent input

This is controlled by `dejitter=True` (default) and `dejitter_period=6` on `MACE4DModel`.

## Timing Log

`model.recon(parallel=True, log_dir="./output/logs")` writes two files. `run_info.txt` is a human-readable summary of the run settings (the model writes its section; the calling script appends dataset and preprocessing settings). `timing_log.csv` has one row per MACE iteration:

```
iteration, agent_0_forward_sec, agent_1_prior_xyt_sec,
agent_2_prior_yzt_sec, agent_3_prior_xzt_sec, iteration_total_sec,
consensus_change_pct
```

`consensus_change_pct` is the relative change of the consensus average between iterations, the convergence measure. With `log_dir=None` (the default) no log files are written.
