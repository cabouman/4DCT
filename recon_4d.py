"""
4D MACE CT Reconstruction — command-line driver.

Typical usage (via shell script):
    bash demo_4d.sh

Or directly:
    python recon_4d.py --data_path /path/to/nsi/dataset

Every parameter is a CLI flag; the defaults are the validated values for the
4DCT phantom dataset.
"""

import os
import sys

# Strip incompatible system CUDA/cuDNN from LD_LIBRARY_PATH before JAX initializes.
# This must run before any JAX import.
if "LD_LIBRARY_PATH" in os.environ and not os.environ.get("_JAX_CLEAN_REEXEC"):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["_JAX_CLEAN_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

import argparse
import time

import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np

from model_4d import MACE4DModel
from utils import construct_time_frames, gen_gif_and_save

if __name__ == "__main__":

    # ── CLI ────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="4D MACE CT Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data and output
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="NSI dataset directory, or a .tgz path/URL to download and extract.",
    )
    parser.add_argument(
        "--download_dir", type=str, default="./data",
        help="Extraction directory used when --data_path is a .tgz.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./output",
        help="Directory for output files (recon, init cache, logs, GIF).",
    )
    # Preprocessing
    parser.add_argument(
        "--downsample_row", type=int, default=1,
        help="Detector row subsampling factor.",
    )
    parser.add_argument(
        "--downsample_column", type=int, default=1,
        help="Detector column subsampling factor.",
    )
    parser.add_argument(
        "--subsample_view_factor", type=int, default=1,
        help="View subsampling factor.",
    )
    parser.add_argument(
        "--sharpness", type=float, default=1.0,
        help="mbirjax sharpness parameter.",
    )
    # Time frames
    parser.add_argument(
        "--angle_span_per_frame", type=float, default=120.0,
        help="Angular span (degrees) covered by each time frame.",
    )
    parser.add_argument(
        "--angle_stride", type=float, default=60.0,
        help="Degrees advanced per frame step (= span - overlap).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Reconstruct only the first N time frames. Omit to use all frames.",
    )
    # MACE
    parser.add_argument(
        "--max_mace_itr", type=int, default=10,
        help="Maximum number of outer MACE iterations.",
    )
    parser.add_argument(
        "--prior_weight", type=float, default=0.5,
        help="Total weight of the three prior agents.",
    )
    parser.add_argument(
        "--rho", type=float, default=0.5,
        help="ADMM step size (Mann iteration parameter).",
    )
    parser.add_argument(
        "--num_prox_iterations", type=int, default=3,
        help="Max prox_map iterations per MACE step.",
    )
    parser.add_argument(
        "--stop_threshold", type=float, default=0.02,
        help="Prox_map convergence threshold.",
    )
    parser.add_argument(
        "--sigma_p", type=float, default=None,
        help="Proximal map sigma. Omit for automatic selection.",
    )
    parser.add_argument(
        "--no_dejitter", action="store_true",
        help="Disable the DCT-I temporal dejitter.",
    )
    # Execution
    parser.add_argument(
        "--serial", action="store_true",
        help="Run the agents sequentially on one device instead of 4 GPUs.",
    )
    parser.add_argument(
        "--device_indices", type=int, nargs=4, default=[0, 1, 2, 3],
        metavar=("FWD", "XYT", "YZT", "XZT"),
        help="GPU indices for the four agents (parallel mode).",
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="0 = silent, 1 = progress, 2 = debug.",
    )
    args = parser.parse_args()

    # ── Resolve dataset ────────────────────────────────────────────────────────
    # A directory is used as is; anything else is downloaded/extracted.
    if os.path.isdir(args.data_path):
        dataset_dir = args.data_path
    else:
        os.makedirs(args.download_dir, exist_ok=True)
        dataset_dir = mj.download_and_extract(args.data_path, args.download_dir)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    init_dir = os.path.join(output_path, "init")
    log_dir = os.path.join(output_path, "logs")

    # CLI angles arrive in degrees; all computation is in radians.
    angle_span_per_frame = np.radians(args.angle_span_per_frame)
    angle_stride         = np.radians(args.angle_stride)
    dejitter_period = int(round(2.0 * np.pi / angle_stride))  # period of the jitter introduced by sinogram gating

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print("\n************** NSI dataset preprocessing **************")
    sino, ct_model = mjp.nsi.get_sino_and_model(
        dataset_dir,
        downsample_factor=[args.downsample_row, args.downsample_column],
        subsample_view_factor=args.subsample_view_factor,
        auto_crop=True,
    )
    ct_model.set_params(sharpness=args.sharpness, positivity_flag=True, verbose=args.verbose)

    # ── Time frame construction ────────────────────────────────────────────────
    print("\n************** Construct time frames **************")
    sino_frames, model_frames = construct_time_frames(
        sino=sino,
        model=ct_model,
        angle_span_per_frame=angle_span_per_frame,
        angle_stride=angle_stride,
    )
    print(f"Total frames: {len(sino_frames)}")
    if args.num_frames is not None:
        sino_frames = sino_frames[:args.num_frames]
        model_frames = model_frames[:args.num_frames]
        print(f"Using first {len(sino_frames)} frames (--num_frames={args.num_frames}).")

    # ── Build model ────────────────────────────────────────────────────────────
    print("\n************** Build 4D MACE model **************")
    model_4d = MACE4DModel(
        sino_list=sino_frames,
        model_list=model_frames,
        prior_weight=args.prior_weight,
        rho=args.rho,
        max_mace_itr=args.max_mace_itr,
        num_prox_iterations=args.num_prox_iterations,
        stop_threshold=args.stop_threshold,
        sigma_p=args.sigma_p,
        dejitter=not args.no_dejitter,
        dejitter_period=dejitter_period,
        verbose=args.verbose,
    )

    # ── Reconstruct ────────────────────────────────────────────────────────────
    print("\n************** Run 4D MACE reconstruction **************")
    time0 = time.time()
    recon_4d = model_4d.recon(
        parallel=not args.serial,
        init_dir=init_dir,
        log_dir=log_dir,
        device_indices=args.device_indices,
    )
    run_time_h = (time.time() - time0) / 3600

    out_path = os.path.join(output_path, f"recon_4d_{run_time_h:.2f}h.npy")
    np.save(out_path, recon_4d)
    print(f"\n[INFO] Total wall time: {run_time_h:.2f} hours.")
    print(f"[INFO] Recon saved to: {os.path.abspath(out_path)}")
    print(f"[INFO] Logs:           {os.path.abspath(log_dir)}")

    with open(os.path.join(log_dir, "run_info.txt"), "a") as f:
        f.write("\n# Script settings (recon_4d.py)\n")
        f.write(f"dataset              = {dataset_dir}\n")
        f.write(f"downsample (row, col) = ({args.downsample_row}, {args.downsample_column})\n")
        f.write(f"subsample_view_factor = {args.subsample_view_factor}\n")
        f.write(f"angle span / stride  = {args.angle_span_per_frame} deg / {args.angle_stride} deg\n")
        f.write(f"num_frames           = {len(sino_frames)}\n")
        f.write(f"sharpness            = {args.sharpness}\n")
        f.write(f"total wall time      = {run_time_h:.2f} h\n")
        f.write(f"recon saved to       = {os.path.abspath(out_path)}\n")

    gif_path = os.path.join(output_path, "recon_4d.gif")
    gen_gif_and_save(recon_4d, gif_path)
    print(f"[INFO] GIF saved to:   {os.path.abspath(gif_path)}")
