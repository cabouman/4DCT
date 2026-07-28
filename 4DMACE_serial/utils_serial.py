"""Simple 4D MACE demo using MBIRJAX cone-beam prox_map and qGGMRF prior denoisers."""

from __future__ import annotations

import time
import threading
import jax.numpy as jnp
import numpy as np
import mbirjax as mj
import os

_THREAD_LOCAL = threading.local()

def truncate_sino_into_time_bins(sino, model, views_per_bin, stride):
    """
    Truncate a sinogram into fixed-size time bins along the view axis (axis=0).

    Args:
        sino (ndarray):
            Input sinogram array with shape (num_views, ...).

        model (mbirjax.ConeBeamModel):
            Fully-built model for the full scan, e.g. from
            mbirjax.preprocess.nsi.get_sino_and_model(...).

        views_per_bin (int):
            Number of views in each time bin.

        stride (int):
            Step size between consecutive bins along the view axis.

    Returns:
        list of tuples:
            Each element is (sino_t, model_t, sl), where each `sino_t` has
            exactly `views_per_bin` views and `model_t` is a per-bin
            ConeBeamModel (built via mbirjax.copy_ct_model with sliced angles
            and recomputed reconstruction geometry).
            Any trailing views that cannot form a full bin are discarded.
    """

    required_params, _, _ = model.get_all_params()
    angles = required_params["angles"]
    num_views = sino.shape[0]

    if views_per_bin <= 0:
        raise ValueError("views_per_bin must be a positive integer.")
    if stride <= 0:
        raise ValueError("slide must be a positive integer.")
    if views_per_bin > num_views:
        raise ValueError("views_per_bin cannot exceed the total number of views.")

    bins = []
    for start in range(0, num_views - views_per_bin + 1, stride):
        end = start + views_per_bin
        sl = slice(start, end)
        sino_t = sino[sl]
        model_t = mj.copy_ct_model(model, new_angles=angles[sl])
        bins.append((sino_t, model_t, sl))

    return bins

def get_qggmrf_denoiser(shape):
    """Return a per-thread cached denoiser to avoid cross-thread state sharing."""
    cache = getattr(_THREAD_LOCAL, "denoiser_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.denoiser_cache = cache
    if shape not in cache:
        cache[shape] = mj.QGGMRFDenoiser(shape)
    return cache[shape]


def normalize_prior_weights(prior_weight):
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = [w for w in prior_weight]
        return [1.0 - sum(prior)] + prior
    w = prior_weight
    return [1.0 - w, w / 3.0, w / 3.0, w / 3.0]

def estimate_sigma_per_hyperplane(x, sigma_noise_floor=1e-6):
    """
    Estimate one sigma value per hyperplane.
    x shape: (num_hyperplanes, dim1, dim2)
    """
    denoiser = get_qggmrf_denoiser(x.shape[1:])
    sigma_list = np.empty(x.shape[0], dtype=np.float32)

    for i in range(x.shape[0]):
        sigma_use = denoiser.estimate_image_noise_std(x[i][:,::4,::4])
        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            sigma_use = 0.0
        sigma_list[i] = sigma_use

    return sigma_list


def qggmrf_hyperplane_denoise(x, sigma_list, sigma_noise_floor=1e-6):
    """
    Denoise a stack of hyperplanes serially using precomputed sigma_list.
    x shape: (num_hyperplanes, dim1, dim2)
    """
    denoiser = get_qggmrf_denoiser(x.shape[1:])
    y = np.empty_like(x)

    for i in range(x.shape[0]):
        sigma_use = sigma_list[i]

        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            y[i] = x[i]
        else:
            y_i, _ = denoiser.denoise(image=x[i], sigma_noise=sigma_use)
            y[i] = np.asarray(y_i)

    return y


def denoiser_wrapper(x, permute_vector, sigma_list):
    """
    Permute 4D volume, denoise the hyperplane stack, then permute back.
    """
    x_perm = np.transpose(x, permute_vector)
    y_perm = qggmrf_hyperplane_denoise(x_perm, sigma_list=sigma_list)
    inv_perm = np.argsort(permute_vector)
    y = np.transpose(y_perm, inv_perm)
    return y


def run_mace_with_models(
    models,
    sino_list,
    weights_list,
    beta,
    max_admm_itr=10,
    rho=0.5,
    forward_num_iterations=3,
    stop_threshold=0.02,
    init_image=None,
    sigma_p=None,
    verbose=1,
    init_save_dir=None
):
    nt = len(sino_list)

    if verbose:
        print(f"[MACE] Start 4D reconstruction with {nt} time bins.")

    if init_image is None:
        if verbose:
            print("[MACE] Computing initial MBIR recon for each time bin...")
        t0 = time.time()
        init_image = np.asarray([
            np.asarray(
                models[t].recon(
                    jnp.asarray(sino_list[t]),
                    weights=jnp.asarray(weights_list[t]),
                    max_iterations=20,
                    stop_threshold_change_pct=stop_threshold,
                )[0])
            for t in range(nt)
        ])
        os.makedirs(init_save_dir, exist_ok=True)
        np.save(os.path.join(init_save_dir, "init_image.npy"), init_image)
        if verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
    else:
        init_image = np.asarray(init_image)
        if verbose:
            print("[MACE] Using provided init_image.")

    # Precompute sigma lists from the initialization
    if verbose:
        print("[MACE] Precomputing sigma_list for each denoising direction...")

    sigma_xyt = estimate_sigma_per_hyperplane(np.transpose(init_image, (3, 0, 1, 2)))

    sigma_yzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (1, 0, 2, 3)))

    sigma_xzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (2, 0, 1, 3)))

    if verbose:
        print("[MACE] sigma_list precomputation done.")

    W = [np.copy(init_image) for _ in range(4)]
    X = [np.copy(init_image) for _ in range(4)]

    for itr in range(max_admm_itr):
        if verbose:
            itr_t0 = time.time()
            print(f"[MACE] Iteration {itr + 1}/{max_admm_itr}")

        if verbose:
            print("[MACE]  Forward agent: cone-beam prox_map for all time bins...")
        X[0] = np.asarray([
            np.asarray(
                models[t].prox_map(
                    prox_input=jnp.asarray(W[0][t]),
                    sinogram=jnp.asarray(sino_list[t]),
                    sigma_prox=sigma_p,
                    weights=jnp.asarray(weights_list[t]),
                    init_recon=jnp.asarray(X[0][t]),
                    max_iterations=forward_num_iterations,
                    stop_threshold_change_pct=stop_threshold,
                )[0]
            )
            for t in range(nt)
        ])

        if verbose:
            print("[MACE]  Prior agent XY-t (fixed z slabs)...")
        X[1] = denoiser_wrapper(W[1], permute_vector=(3, 0, 1, 2), sigma_list=sigma_xyt)

        if verbose:
            print("[MACE]  Prior agent YZ-t (fixed row slabs)...")
        X[2] = denoiser_wrapper(W[2], permute_vector=(1, 0, 2, 3), sigma_list=sigma_yzt)

        if verbose:
            print("[MACE]  Prior agent XZ-t (fixed col slabs)...")
        X[3] = denoiser_wrapper(W[3], permute_vector=(2, 0, 1, 3), sigma_list=sigma_xzt,)

        if verbose:
            print("[MACE]  Consensus / ADMM update...")
        z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))

        for k in range(4):
            W[k] = W[k] + 2.0 * rho * (z - X[k])

        if verbose:
            print(f"[MACE] Iteration {itr + 1} done in {time.time() - itr_t0:.2f} sec.")

    if verbose:
        print("[MACE] Reconstruction complete.")

    return sum(beta[k] * X[k] for k in range(4))


def mace4d_from_cone_beam_params(
    sino_list,
    model_list,
    weight_type="transmission_root",
    prior_weight=0.5,
    max_admm_itr=10,
    rho=0.5,
    forward_num_iterations=3,
    stop_threshold=0.02,
    init_image=None,
    sigma_p=None,
    verbose=1,
    init_save_dir=None):
    if verbose:
        print("[MACE] Building weights for each time bin...")

    weights_list = [mj.gen_weights(jnp.asarray(s), weight_type=weight_type) for s in sino_list]

    if verbose:
        print(f"[MACE] Built weights for {len(weights_list)} time bins.")

    recon_4d = run_mace_with_models(
        models=model_list,
        sino_list=sino_list,
        weights_list=weights_list,
        beta=normalize_prior_weights(prior_weight),
        max_admm_itr=max_admm_itr,
        rho=rho,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        init_image=init_image,
        sigma_p=sigma_p,
        verbose=verbose,
        init_save_dir=init_save_dir
    )
    return recon_4d