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
from utils import construct_time_frames, gen_gif_and_save

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
        "--angle_span_per_recon", type=float, default=120.0,
        help="Angular span (degrees) covered by each time frame.",
    )
    parser.add_argument(
        "--angle_advancing", type=float, default=60.0,
        help="Degrees advanced per frame step (= span - overlap).",
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
        "--num_frames", type=int, default=None,
        help="Reconstruct only the first N time frames. Omit to use all frames.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="INIT_PATH",
        help="Path to a saved init_image.npy to skip per-frame initialization.",
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
    angle_span_per_recon = args.angle_span_per_recon  # Angular span (degrees) covered by each time frame.
    angle_advancing      = args.angle_advancing  # Degrees advanced per frame step.
    dejitter_period = int(round(360.0 / angle_advancing))  # period of the jitter introduced by sinogram gating

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print("\n************** NSI dataset preprocessing **************")
    downsample_rate = [args.downsample_row, args.downsample_column]
    sino, ct_model = mjp.nsi.get_sino_and_model(
        args.data_path,
        downsample_factor=downsample_rate,
        subsample_view_factor=args.subsample_view_factor,
        auto_crop=True,
    )
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)

    # ── Time frame construction ────────────────────────────────────────────────
    print("\n************** Construct time frames **************")
    sino_frames, model_frames = construct_time_frames(
        sino=sino,
        model=ct_model,
        angle_span_per_recon=angle_span_per_recon,
        angle_advancing=angle_advancing,
    )
    print(f"Total frames: {len(sino_frames)}")
    if args.num_frames is not None:
        sino_frames = sino_frames[:args.num_frames]
        model_frames = model_frames[:args.num_frames]
        print(f"Using first {len(sino_frames)} frames (--num_frames={args.num_frames}).")

    # ── Init image (resume or fresh) ───────────────────────────────────────────
    init_image = None
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume path does not exist: {args.resume}")
        print(f"[INFO] Loading saved init_image from {args.resume}")
        init_image = np.load(args.resume)

    # ── Build model ────────────────────────────────────────────────────────────
    print("\n************** Build 4D MACE model **************")
    model_4d = MACE4DModel(
        sino_list=sino_frames,
        model_list=model_frames,
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

    gif_path = os.path.join(output_path, "recon_4d.gif")
    gen_gif_and_save(recon_4d, gif_path)
    print(f"[INFO] GIF saved to:   {os.path.abspath(gif_path)}")
