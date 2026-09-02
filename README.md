# 4DCT MACE Recon

4D CT reconstruction using MACE with a cone-beam prox_map forward agent and three qGGMRF prior agents across XY-t, YZ-t, and XZ-t hyperplanes. Designed for the Eli Lilly 4DCT phantom dataset acquired on NSI hardware.

## Where the implementation lives

The reconstruction code is now part of **mbirjax**, as `mj.MACE4DModel` plus the supporting
utilities `mj.construct_time_frames` and `mj.construct_time_frame_models`. This repository
keeps the Lilly-specific driver (`recon_4d.py`, `demo_4d.sh`) and the operator documentation.

`mace4d.py` and `tests/` are the pre-merge standalone copy, kept only until the mbirjax
branch lands on `main`. Nothing in this repository imports them any more.

## Dependencies

Requires an `mbirjax` that includes `MACE4DModel` (branch `4DCT_for_merging` or later), plus
`imageio` and `matplotlib` for the output GIF.

## Quick Start (Lilly)

Edit `DATA_PATH` in `demo_4d.sh` to point to your extracted NSI dataset, then:

```bash
bash demo_4d.sh
```

Output is written to `./output/`:
- `recon_4d_<time>h.npy` — reconstructed 4D volume, shape `(nt, nx, ny, nz)`
- `init/init_recon.npy` — per-frame MBIR initialization (saved for potential reuse)
- `logs/run_info.txt` — human-readable summary of all run settings
- `logs/timing_log.csv` — per-iteration agent timing and consensus change
- `recon_4d.gif` — the middle x slice playing over time (`--gif_vmax` sets the display range)

On a second run with the same settings, the saved initialization in `output/init/` is detected and reused automatically; if its shape does not match the run, it is recomputed with a warning.

## Layout

```
4DCT/
├── demo_4d.sh            # shell entry point — edit DATA_PATH and run
├── recon_4d.py           # command-line reconstruction driver (all params as flags)
├── mace4d.py             # pre-merge standalone copy, superseded by mbirjax.MACE4DModel
├── plans/
│   ├── mbirjax_merge_plan.md  # the merge design and its decisions
│   └── lilly_interface.md     # parameter reference for Lilly operators
└── tests/                # pre-merge tests, superseded by mbirjax tests/test_mace4d.py
```

## File Roles

| File | Purpose |
|------|---------|
| `demo_4d.sh` | Shell entry point. Edit `DATA_PATH`; everything else has sensible defaults. |
| `recon_4d.py` | Reconstruction driver. Every parameter is a CLI flag with a validated default: data path, downsampling, frame geometry, MACE hyperparameters, weighting, execution mode (`--serial`; by default all visible GPUs are used — restrict with `CUDA_VISIBLE_DEVICES`). |

## MACE4DModel Interface

```python
import mbirjax as mj
import mbirjax.preprocess as mjp

# 1. Preprocess with mbirjax
sino, ct_model = mjp.nsi.get_sino_and_model(dataset_dir, auto_crop=True)
ct_model.set_params(sharpness=1.0, positivity_flag=True)

# 2. Build the model: 6 frames per rotation, each view shared by 2 frames.
#    The frame structure comes from the model's angles, so no data is needed here.
mace_model = mj.MACE4DModel(ct_model, frames_per_rotation=6, frame_overlap_factor=2.0)
mace_model.set_params(mace_prior_weight=0.5, rho_mann=0.5)

# 3. Reconstruct. transmission_root is the validated weighting for 4D data; the model
#    itself defaults to unit weights, following TomographyModel.recon.
weights = mj.gen_weights(sino, weight_type='transmission_root')
recon_4d, recon_dict = mace_model.recon(sino, weights=weights, max_iterations=10,
                                        init_dir="./output/init")
```

The per-frame models and view slices are available straight from the constructor as
`mace_model.model_list` and `mace_model.view_slices`, so a single frame can be inspected
before committing to a full 4D run. For a standalone per-frame workflow, use
`mj.construct_time_frames(sino, ct_model)` and reconstruct one frame directly.

## Multi-GPU Architecture

Each MACE iteration is a set of independent tasks: one cone-beam `prox_map` per
time frame and one batched qGGMRF denoise per hyperplane orientation (XY-t,
YZ-t, XZ-t). One worker thread per visible GPU executes the tasks; a fixed
least-loaded assignment, computed once, maps every task to one GPU for the
whole run. The per-frame initialization uses the same workers and map.

**GPU pinning**: every `ConeBeamModel` and `QGGMRFDenoiser` is pinned to exactly one GPU via `configure_devices([device])`, and each frame's sinogram and weights live on that GPU for the whole run. Pinning prevents the NCCL clique deadlock that occurs when mbirjax's auto-sharding builds a multi-GPU Mesh inside concurrent threads.

Devices are selected with `mace_model.set_device_pool(...)`: `None` (or never calling it)
uses all visible GPUs, `1` forces the serial path, and an explicit list pins exactly those
devices. One device runs the same tasks inline with no threads.

## DCT-I Temporal Dejitter

The dejitter step removes periodic gating jitter (one period per rotation) by zeroing the corresponding DCT-I frequency bands. It is applied:
- After the forward agent output
- Before each prior agent input

This is controlled by `set_params(dejitter=True)` (the default) and the constructor's
`frames_per_rotation`, which is also the jitter period.

## Timing Log

`mace_model.recon(..., log_dir="./output/logs")` writes three files. `task_log.csv` has one row per task (iteration, kind, index, device, start, end). `run_info.txt` is a human-readable summary of the run settings (the model writes its section; the calling script appends dataset and preprocessing settings). `timing_log.csv` has one row per MACE iteration:

```
iteration, prox_total_sec, denoise_total_sec, makespan_sec,
iteration_total_sec, consensus_change_pct
```

`consensus_change_pct` is the relative change of the consensus average between iterations, the convergence measure, and the quantity `stop_threshold_change_pct` tests against. With `log_dir=None` (the default) no log files are written; the same settings and per-iteration timing are always available in the returned `recon_dict`.
