# Lilly Interface — `recon_4d.py`

Entry point for Lilly production runs. Launched via `bash demo_4d.sh` or directly as
`python recon_4d.py --data_path <path>`.

---

## Parameters Lilly needs to set

### In `demo_4d.sh`

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/` | Path to the extracted NSI dataset directory (must contain a `.nsipro` file) |
| `OUTPUT_PATH` | `./output` | Directory for all outputs (recon `.npy`, init image, timing CSV) |

### CLI flags (passed to `recon_4d.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--data_path` | *(required)* | Same as `DATA_PATH` above |
| `--downsample_row` | `1` | Detector row subsampling factor; increase to trade resolution for speed |
| `--downsample_column` | `1` | Detector column subsampling factor; increase to trade resolution for speed |
| `--subsample_view_factor` | `1` | View subsampling factor; increase to use fewer projection angles |
| `--angle_span_per_frame` | `120.0` | Degrees covered per time frame |
| `--angle_stride` | `60.0` | Degrees advanced per frame step (= span − overlap) |
| `--max_mace_itr` | `10` | Number of outer MACE iterations |
| `--output_path` | `./output` | Output directory |
| `--num_frames` | *(all frames)* | Reconstruct only the first N time frames |

---

## Advanced CLI flags (defaults are the validated values; Lilly normally leaves them alone)

| Flag | Default | Meaning |
|---|---|---|
| `--prior_weight` | `0.5` | Total weight of the three prior agents (forward agent gets 1 − w) |
| `--rho` | `0.5` | ADMM step size (Mann iteration parameter) |
| `--num_prox_iterations` | `3` | Max prox_map iterations per MACE step |
| `--stop_threshold` | `0.02` | Prox_map convergence threshold (% change) |
| `--sharpness` | `1.0` | Sharpness parameter passed to `ct_model` |
| `--sigma_p` | *(auto)* | Proximal sigma; omit for automatic selection |
| `--no_dejitter` | *(off)* | Disables the DCT-I temporal dejitter |
| `--serial` | *(off)* | Run all tasks on one device. By default all visible GPUs are used; restrict with `CUDA_VISIBLE_DEVICES` |
| `--download_dir` | `./data` | Extraction directory when `--data_path` is a `.tgz` |
| `--verbose` | `1` | 0 = silent, 1 = progress, 2 = debug |

---

## Derived parameters (not settable)

### Scan geometry — derived automatically

| Parameter | Value | How it's set                                                                     |
|---|---|----------------------------------------------------------------------------------|
| `views_per_frame` | derived | `round(angle_span_per_frame / angle_step)` inside `construct_time_frames()`; angle_step comes from the model's view spacing |
| `stride` | derived | `round(angle_stride / angle_step)` inside `construct_time_frames()`              |
| `dejitter_period` | derived | `round(2π / angle_stride)` = 6 for 60° stride                                    |

`angle_span_per_frame` and `angle_stride` are CLI flags in degrees (defaults 120° and 60°); internally all angles are radians. The sinogram weighting scheme is fixed at `"transmission_root"` (default in `MACE4DModel`).

> **Slurm note (Magtrain cluster):** Always request 4 GPUs when submitting the job, e.g. `--gres=gpu:4`.

---

## Outputs (written to `output_path/`)

| File | Description |
|---|---|
| `recon_4d_<time>h.npy` | Final 4D reconstruction, shape `(nt, nx, ny, nz)` |
| `init/init_image.npy` | Per-frame MBIR initialization; reused automatically on re-runs when its shape matches |
| `logs/run_info.txt` | Human-readable summary of all run settings |
| `logs/timing_log.csv` | Per-iteration prox/denoise/makespan times and consensus change (%) |
| `logs/task_log.csv` | One row per task: iteration, kind, index, device, start, end |
