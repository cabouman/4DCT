"""
4D MACE CT Reconstruction — development / exploration script.

Edit the parameter blocks below before each run. Unlike Lilly_recon.py, all
hyperparameters are directly accessible as Python variables. Toggle
USE_SAVED_INIT_IMAGE to skip the slow per-frame initialization step on
repeated runs against the same dataset.
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

import time

import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np

from model_4d import MACE4DModel
from utils import construct_time_frames, gen_gif_and_save

if __name__ == "__main__":

    # ── Paths ──────────────────────────────────────────────────────────────────
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)
    download_dir = "./data"
    os.makedirs(download_dir, exist_ok=True)

    # ── Init image ─────────────────────────────────────────────────────────────
    # Set to True to reuse a previously saved init (fast for repeated runs).
    # The expected path is output_path/init/init_image.npy.
    USE_SAVED_INIT_IMAGE = False
    if USE_SAVED_INIT_IMAGE:
        init_image_path = os.path.join(output_path, "init", "init_image.npy")
        init_image = np.load(init_image_path)
    else:
        init_image = None

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset_url = "/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024.tgz"
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    downsample_rate = [1, 1]
    subsample_view_factor = 1
    sino_auto_cropping = True
    sharpness = 1.0
    verbose = 1

    print("\n************** NSI dataset preprocessing **************")
    sino, ct_model = mjp.nsi.get_sino_and_model(
        dataset_dir,
        downsample_factor=downsample_rate,
        subsample_view_factor=subsample_view_factor,
        auto_crop=sino_auto_cropping,
    )
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)

    # ── Time frame construction ────────────────────────────────────────────────
    angle_span_per_recon = 120.0   # degrees covered per time frame
    angle_overlapping    = 60.0    # degrees of overlap between frames
    angle_advancing = angle_span_per_recon - angle_overlapping  # degrees advanced per frame step
    dejitter_period = int(round(360.0 / angle_advancing))  # period of the jitter introduced by sinogram gating

    time_range = slice(0, -1)

    print("\n************** Construct time frames **************")
    sino_frames, model_frames = construct_time_frames(
        sino=sino,
        model=ct_model,
        angle_span_per_recon=angle_span_per_recon,
        angle_advancing=angle_advancing,
    )
    sino_frames = sino_frames[time_range]
    model_frames = model_frames[time_range]
    print(f"Total frames: {len(sino_frames)}")

    # ── MACE hyperparameters ───────────────────────────────────────────────────
    prior_weight = 0.5          # float or 3-list [xyt_w, yzt_w, xzt_w]
    rho = 0.5                   # ADMM step size
    max_mace_itr = 10           # outer MACE iterations
    forward_num_iterations = 3  # prox_map iterations per step
    stop_threshold = 0.02       # prox_map convergence threshold
    sigma_p = None              # proximal sigma; None = auto
    dejitter = True             # DCT-I temporal dejitter inside agents

    # ── Execution mode ─────────────────────────────────────────────────────────
    parallel = True             # False → serial mode (single device)
    device_indices = [0, 1, 2, 3]  # [forward, prior_xyt, prior_yzt, prior_xzt]

    # ── Build model ────────────────────────────────────────────────────────────
    print("\n************** Build 4D MACE model **************")
    model_4d = MACE4DModel(
        sino_list=sino_frames,
        model_list=model_frames,
        prior_weight=prior_weight,
        rho=rho,
        max_mace_itr=max_mace_itr,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        sigma_p=sigma_p,
        verbose=verbose,
        dejitter=dejitter,
        dejitter_period=dejitter_period,
    )

    init_save_dir = os.path.join(output_path, "init")
    timing_log_path = os.path.join(output_path, "timing_log.csv")

    # ── Reconstruct ────────────────────────────────────────────────────────────
    print("\n************** Run 4D MACE reconstruction **************")
    time0 = time.time()
    recon_4d = model_4d.recon(
        init_image=init_image,
        parallel=parallel,
        init_save_dir=init_save_dir,
        timing_log_path=timing_log_path,
        device_indices=device_indices,
    )
    run_time_h = (time.time() - time0) / 3600

    out_path = os.path.join(output_path, f"recon_4d_{run_time_h:.2f}h.npy")
    np.save(out_path, recon_4d)
    print(f"\n[MACE] Total wall time: {run_time_h:.2f} hours.")
    print(f"[MACE] Recon saved to:  {os.path.abspath(out_path)}")

    gif_path = os.path.join(output_path, "recon_4d.gif")
    gen_gif_and_save(recon_4d, gif_path)
    print(f"[MACE] GIF saved to:    {os.path.abspath(gif_path)}")
