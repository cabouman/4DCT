# Lilly Interface — `Lilly_recon.py`

Entry point for Lilly production runs. Launched via `bash test_script_4D.sh` or directly as
`python Lilly_recon.py --data_path <path>`.

---

## Parameters Lilly needs to set

### In `test_script_4D.sh`

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/` | Path to the extracted NSI dataset directory (must contain a `.nsipro` file) |
| `OUTPUT_PATH` | `./output` | Directory for all outputs (recon `.npy`, init image, timing CSV) |

### CLI flags (passed to `Lilly_recon.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--data_path` | *(required)* | Same as `DATA_PATH` above |
| `--downsample` | `1` | Detector pixel subsampling factor (rows and channels); increase to trade resolution for speed |
| `--subsample_view_factor` | `1` | View subsampling factor; increase to use fewer projection angles |
| `--max_mace_itr` | `10` | Number of outer MACE iterations |
| `--output_path` | `./output` | Output directory |
| `--resume` | *(flag, off by default)* | If set, reloads a previously saved `init_image.npy` from `output_path/init/` instead of recomputing it |

---

## Parameters fixed by us (not exposed to Lilly)

### Scan geometry — derived automatically from the `.nsipro` file

| Parameter | Value | How it's set                                                                     |
|---|---|----------------------------------------------------------------------------------|
| `angle_span_per_recon` | `120.0°` | Degrees covered per time bin — hardcoded in script for now                       |
| `angle_overlapping` | `60.0°` | Angular overlap between consecutive bins — hardcoded in script for now           |
| `angle_march` | derived | `angle_span_per_recon - angle_overlapping` = 60° (degrees advanced per bin step) |
| `views_per_bin` | derived | `round(angle_span / angle_step)` via `compute_bin_params()`                      |
| `stride` | derived | `round(angle_march / angle_step)` via `compute_bin_params()`                     |
| `dejitter_period` | derived | `round(360 / angle_march)` = 6 for 60° march                                     |

### MACE algorithm

| Parameter | Value | Meaning |
|---|---|---|
| `prior_weight` | `0.5` | Weight split between forward agent (0.5) and three prior agents (0.5/3 each) |
| `rho` | `0.5` | ADMM step size (Mann iteration parameter) |
| `forward_num_iterations` | `3` | Max prox_map iterations per MACE step |
| `stop_threshold` | `0.02` | Prox_map convergence threshold (% change) |
| `weight_type` | `"transmission_root"` | Sinogram weighting scheme (default in `MACE4DModel`) |
| `sigma_p` | `None` | Proximal sigma — `None` lets mbirjax choose automatically |
| `sharpness` | `1.0` | Sharpness parameter passed to `ct_model` |
| `dejitter` | `True` | DCT-I temporal dejitter applied inside each MACE agent |

### Execution

| Parameter | Value | Meaning |
|---|---|---|
| `parallel` | `True` | 4-GPU ThreadPoolExecutor mode (requires ≥4 GPUs) |
| `device_indices` | `[0,1,2,3]` | GPU assignment: `[forward, prior_xyt, prior_yzt, prior_xzt]` |

> **Slurm note (Magtrain cluster):** Always request 4 GPUs when submitting the job, e.g. `--gres=gpu:4`.

---

## Outputs (written to `output_path/`)

| File | Description |
|---|---|
| `recon_4d_<time>h.npy` | Final 4D reconstruction, shape `(nt, nx, ny, nz)` |
| `init/init_image.npy` | Per-bin MBIR initialization (reused with `--resume`) |
| `timing_log.csv` | Per-iteration wall time for each MACE agent |
