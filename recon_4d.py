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

from mace4d import MACE4DModel, construct_time_frames, gen_gif_and_save


def parse_args():
    """Command-line interface. Defaults are the validated values for the 4DCT phantom."""
    parser = argparse.ArgumentParser(
        description="4D MACE CT Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("data and output")
    g.add_argument("--data_path", type=str, required=True,
                   help="NSI dataset directory, or a .tgz path/URL to download and extract.")
    g.add_argument("--download_dir", type=str, default="./data",
                   help="Extraction directory used when --data_path is a .tgz.")
    g.add_argument("--output_path", type=str, default="./output",
                   help="Directory for output files (recon, init cache, logs, GIF).")

    g = parser.add_argument_group("preprocessing")
    g.add_argument("--downsample_row", type=int, default=1, help="Detector row subsampling factor.")
    g.add_argument("--downsample_column", type=int, default=1, help="Detector column subsampling factor.")
    g.add_argument("--subsample_view_factor", type=int, default=1, help="View subsampling factor.")
    g.add_argument("--sharpness", type=float, default=1.0, help="mbirjax sharpness parameter.")

    g = parser.add_argument_group("time frames")
    g.add_argument("--frames_per_rotation", type=int, default=6,
                   help="Time frames per full 360-degree rotation.")
    g.add_argument("--frame_overlap_factor", type=float, default=2.0,
                   help="Number of frames that share any given view. Each frame "
                        "spans frame_overlap_factor * (360 / frames_per_rotation) degrees.")
    g.add_argument("--num_frames", type=int, default=None,
                   help="Reconstruct only the first N time frames. Omit (default) to use all frames. "
                        "If N exceeds the total frame count, all frames are used.")

    g = parser.add_argument_group("MACE algorithm")
    g.add_argument("--max_mace_itr", type=int, default=10, help="Maximum number of outer MACE iterations.")
    g.add_argument("--mace_prior_weight", type=float, default=0.5, help="Total weight of the three prior agents.")
    g.add_argument("--rho_mann", type=float, default=0.5, help="Mann iteration step size (ADMM rho).")
    g.add_argument("--prox_num_iterations", type=int, default=3, help="Max prox_map iterations per MACE step.")
    g.add_argument("--prox_stop_threshold", type=float, default=0.02, help="Prox_map convergence threshold.")
    g.add_argument("--sigma_p", type=float, default=None, help="Proximal map sigma. Omit for automatic selection.")
    g.add_argument("--no_dejitter", action="store_true", help="Disable the DCT-I temporal dejitter.")

    g = parser.add_argument_group("execution")
    g.add_argument("--serial", action="store_true",
                   help="Run all tasks on one device (alias for one-device mode). "
                        "By default all visible GPUs are used; restrict them with CUDA_VISIBLE_DEVICES.")
    g.add_argument("--verbose", type=int, default=1, help="0 = silent, 1 = progress, 2 = debug.")

    return parser.parse_args()


def resolve_dataset(args):
    """Return the dataset directory; a non-directory data_path is downloaded/extracted."""
    if os.path.isdir(args.data_path):
        return args.data_path
    os.makedirs(args.download_dir, exist_ok=True)
    return mj.download_and_extract(args.data_path, args.download_dir)


def append_run_info(log_dir, args, dataset_dir, num_frames, run_time_h, out_path):
    """Append the script settings to the run_info.txt started by MACE4DModel.recon()."""
    with open(os.path.join(log_dir, "run_info.txt"), "a") as f:
        f.write("\n# Script settings (recon_4d.py)\n")
        f.write(f"dataset              = {dataset_dir}\n")
        f.write(f"downsample (row, col) = ({args.downsample_row}, {args.downsample_column})\n")
        f.write(f"subsample_view_factor = {args.subsample_view_factor}\n")
        f.write(f"frames_per_rotation  = {args.frames_per_rotation}\n")
        f.write(f"frame_overlap_factor = {args.frame_overlap_factor}\n")
        f.write(f"num_frames           = {num_frames}\n")
        f.write(f"sharpness            = {args.sharpness}\n")
        f.write(f"total wall time      = {run_time_h:.2f} h\n")
        f.write(f"recon saved to       = {out_path}\n")


def main():
    args = parse_args()
    dataset_dir = resolve_dataset(args)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    init_dir = os.path.join(output_path, "init")
    log_dir = os.path.join(output_path, "logs")

    print("\n************** NSI dataset preprocessing **************")
    sino, ct_model = mjp.nsi.get_sino_and_model(
        dataset_dir,
        downsample_factor=[args.downsample_row, args.downsample_column],
        subsample_view_factor=args.subsample_view_factor,
        auto_crop=True,
    )
    ct_model.set_params(sharpness=args.sharpness, positivity_flag=True, verbose=args.verbose)

    print("\n************** Construct time frames **************")
    sino_frames, model_frames = construct_time_frames(
        sino=sino,
        model=ct_model,
        frames_per_rotation=args.frames_per_rotation,
        frame_overlap_factor=args.frame_overlap_factor,
    )
    print(f"Total frames: {len(sino_frames)}")
    if args.num_frames is not None:
        if args.num_frames >= len(sino_frames):
            print(f"[INFO] --num_frames={args.num_frames} >= total frames ({len(sino_frames)}); reconstructing all frames.")
        else:
            sino_frames = sino_frames[:args.num_frames]
            model_frames = model_frames[:args.num_frames]
            print(f"Using first {len(sino_frames)} frames (--num_frames={args.num_frames}).")

    print("\n************** Build 4D MACE model **************")
    mace_model = MACE4DModel(
        sino_list=sino_frames,
        model_list=model_frames,
        mace_prior_weight=args.mace_prior_weight,
        rho_mann=args.rho_mann,
        max_mace_itr=args.max_mace_itr,
        prox_num_iterations=args.prox_num_iterations,
        prox_stop_threshold=args.prox_stop_threshold,
        sigma_p=args.sigma_p,
        dejitter=not args.no_dejitter,
        frames_per_rotation=args.frames_per_rotation,
        verbose=args.verbose,
    )

    print("\n************** Run 4D MACE reconstruction **************")
    time0 = time.time()
    recon_4d = mace_model.recon(
        devices=1 if args.serial else None,
        init_dir=init_dir,
        log_dir=log_dir,
    )
    run_time_h = (time.time() - time0) / 3600

    out_path = os.path.abspath(os.path.join(output_path, f"recon_4d_{run_time_h:.2f}h.npy"))
    np.save(out_path, recon_4d)
    print(f"\n[INFO] Total wall time: {run_time_h:.2f} hours.")
    print(f"[INFO] Recon saved to: {out_path}")
    print(f"[INFO] Logs:           {os.path.abspath(log_dir)}")

    append_run_info(log_dir, args, dataset_dir, len(sino_frames), run_time_h, out_path)

    gif_path = os.path.join(output_path, "recon_4d.gif")
    gen_gif_and_save(recon_4d, gif_path)
    print(f"[INFO] GIF saved to:   {os.path.abspath(gif_path)}")


if __name__ == "__main__":
    main()
