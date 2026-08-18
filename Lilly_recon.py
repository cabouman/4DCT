"""
4D MACE CT Reconstruction — Lilly production script.

Typical usage (via shell script):
    bash test_script_4D.sh

Or directly:
    python Lilly_recon.py --data_path /path/to/nsi/dataset

The algorithmic hyperparameters (prior_weight, rho, etc.) are fixed below and
not exposed as CLI flags; they represent validated defaults for the 4DCT phantom
dataset. Contact the development team before changing them.
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
from utils import truncate_sino_into_time_bins, compute_bin_params

if __name__ == "__main__":

    # ── CLI ────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="4D MACE CT Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the extracted NSI dataset directory.",
    )
    parser.add_argument(
        "--downsample", type=int, default=1,
        help="Detector subsampling factor (rows and channels).",
    )
    parser.add_argument(
        "--subsample_view_factor", type=int, default=1,
        help="View subsampling factor.",
    )
    parser.add_argument(
        "--max_mace_itr", type=int, default=10,
        help="Maximum number of outer MACE iterations.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./output",
        help="Directory for output files (recon, init_image, timing log).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load a previously saved init_image from output_path/init/ if found.",
    )
    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"--data_path does not exist or is not a directory: {args.data_path}"
        )

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    init_save_dir = os.path.join(output_path, "init")
    timing_log_path = os.path.join(output_path, "timing_log.csv")

    # ── Fixed algorithmic hyperparameters ──────────────────────────────────────
    sharpness = 1.0
    prior_weight = 0.5
    rho = 0.5
    forward_num_iterations = 3
    stop_threshold = 0.02
    verbose = 1
    angle_span_per_recon = 120.0   # degrees covered per time bin
    angle_overlapping    = 60.0    # degrees of overlap between bins
    angle_march = angle_span_per_recon - angle_overlapping  # degrees advanced per bin step
    dejitter_period = int(round(360.0 / angle_march))  # period of the jitter introduced by sinogram gating

    views_per_bin, stride = compute_bin_params(args.data_path, angle_span_per_recon, angle_overlapping)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print("\n************** NSI dataset preprocessing **************")
    downsample_rate = [args.downsample, args.downsample]
    sino, ct_model = mjp.nsi.get_sino_and_model(
        args.data_path,
        downsample_factor=downsample_rate,
        subsample_view_factor=args.subsample_view_factor,
        auto_crop=True,
    )
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)

    # ── Time bin splitting ─────────────────────────────────────────────────────
    print("\n************** Split into time bins **************")
    bins = truncate_sino_into_time_bins(
        sino=sino,
        model=ct_model,
        views_per_bin=views_per_bin,
        stride=stride,
    )
    print(f"Total bins: {len(bins)}")

    sino_list = [b[0] for b in bins]
    model_list = [b[1] for b in bins]

    # ── Init image (resume or fresh) ───────────────────────────────────────────
    init_image = None
    init_image_path = os.path.join(init_save_dir, "init_image.npy")
    if args.resume and os.path.isfile(init_image_path):
        print(f"[INFO] Loading saved init_image from {init_image_path}")
        init_image = np.load(init_image_path)

    # ── Build model ────────────────────────────────────────────────────────────
    print("\n************** Build 4D MACE model **************")
    model_4d = MACE4DModel(
        sino_list=sino_list,
        model_list=model_list,
        prior_weight=prior_weight,
        rho=rho,
        max_mace_itr=args.max_mace_itr,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        verbose=verbose,
        dejitter_period=dejitter_period,
    )

    # ── Reconstruct ────────────────────────────────────────────────────────────
    print("\n************** Run 4D MACE reconstruction **************")
    time0 = time.time()
    recon_4d = model_4d.recon(
        init_image=init_image,
        parallel=True,
        init_save_dir=init_save_dir,
        timing_log_path=timing_log_path,
    )
    run_time_h = (time.time() - time0) / 3600

    out_path = os.path.join(output_path, f"recon_4d_{run_time_h:.2f}h.npy")
    np.save(out_path, recon_4d)
    print(f"\n[INFO] Total wall time: {run_time_h:.2f} hours.")
    print(f"[INFO] Recon saved to: {os.path.abspath(out_path)}")
    print(f"[INFO] Timing log:     {os.path.abspath(timing_log_path)}")
