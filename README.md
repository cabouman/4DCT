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
| `recon_4d.py` | Reconstruction driver. Every parameter is a CLI flag with a validated default: data path, downsampling, frame geometry, MACE hyperparameters, execution mode (`--serial`; by default all visible GPUs are used — restrict with `CUDA_VISIBLE_DEVICES`). |
| `mace4d.py` | The complete 4D functional block: `MACE4DModel` (`model.recon(devices=None)` uses all visible GPUs; `devices=1` is the serial path) plus time-frame construction, DCT-I dejitter, hyperplane-denoiser helpers, and the GIF writer. |

## MACE4DModel Interface

```python
from mace4d import MACE4DModel, construct_time_frames

# 1. Preprocess with mbirjax
sino, ct_model = mjp.nsi.get_sino_and_model(dataset_dir, ...)
ct_model.set_params(sharpness=1.0, positivity_flag=True)

# 2. Construct time frames: 6 frames per rotation, each view shared by 2 frames
sino_frames, model_frames = construct_time_frames(sino, ct_model,
                                                  frames_per_rotation=6,
                                                  frame_overlap_factor=2.0)

# 3. Build model and reconstruct
mace_model = MACE4DModel(sino_frames, model_frames, mace_prior_weight=0.5, max_mace_itr=10)
recon_4d = mace_model.recon(init_dir="./output/init")  # all visible GPUs
```

## Multi-GPU Architecture

Each MACE iteration is a set of independent tasks: one cone-beam `prox_map` per
time frame and one batched qGGMRF denoise per hyperplane orientation (XY-t,
YZ-t, XZ-t). One worker thread per visible GPU executes the tasks; a fixed
least-loaded assignment, computed once, maps every task to one GPU for the
whole run. The per-frame initialization uses the same workers and map.

**GPU pinning**: every `ConeBeamModel` and `QGGMRFDenoiser` is pinned to exactly one GPU via `configure_devices([device])`, and each frame's sinogram and weights live on that GPU for the whole run. Pinning prevents the NCCL clique deadlock that occurs when mbirjax's auto-sharding builds a multi-GPU Mesh inside concurrent threads.

Any number of visible GPUs works; one device (`devices=1`, or CPU-only) runs the same tasks inline with no threads.

## DCT-I Temporal Dejitter

The dejitter step removes periodic gating jitter (one period per rotation) by zeroing the corresponding DCT-I frequency bands. It is applied:
- After the forward agent output
- Before each prior agent input

This is controlled by `dejitter=True` (default) and `frames_per_rotation=6` on `MACE4DModel`.

## Timing Log

`model.recon(log_dir="./output/logs")` writes three files. `task_log.csv` has one row per task (iteration, kind, index, device, start, end). `run_info.txt` is a human-readable summary of the run settings (the model writes its section; the calling script appends dataset and preprocessing settings). `timing_log.csv` has one row per MACE iteration:

```
iteration, prox_total_sec, denoise_total_sec, makespan_sec,
iteration_total_sec, consensus_change_pct
```

`consensus_change_pct` is the relative change of the consensus average between iterations, the convergence measure. With `log_dir=None` (the default) no log files are written.
