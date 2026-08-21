"""
Shared utilities for 4D MACE reconstruction.

Merges utils_serial.py and utils_multi_threads.py. All public functions are
device-aware so they work in both serial (single device) and parallel (one
device per agent) contexts.
"""
from __future__ import annotations

import threading

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax as mj
import numpy as np
from scipy.fft import dct, idct

# Thread-local denoiser cache: key = (shape, device), value = QGGMRFDenoiser.
# Ensures no denoiser instance is shared across threads (critical for multi-GPU).
_THREAD_LOCAL = threading.local()


# ---------------------------------------------------------------------------
# Time frame construction
# ---------------------------------------------------------------------------

def construct_time_frames(sino, model, angle_span_per_frame, angle_stride):
    """
    Split a full sinogram into overlapping fixed-size time frames.

    The number of views per frame and the stride between frames are derived
    from the model's angle spacing, so they stay correct under view subsampling.

    Parameters
    ----------
    sino : ndarray, shape (num_views, det_rows, det_cols)
    model : mbirjax.ConeBeamModel
        Fully-built model for the full scan.
    angle_span_per_frame : float
        Angular span (radians) covered by each time frame.
    angle_stride : float
        Radians advanced per frame step.

    Returns
    -------
    sino_frames : list of ndarray
        Each covers angle_span_per_frame radians of views. Trailing views that
        cannot form a full frame are discarded.
    model_frames : list of mbirjax.ConeBeamModel
        Per-frame models built via mj.copy_ct_model.
    """
    required_params, _, _ = model.get_all_params()
    angles = required_params["angles"]
    num_views = sino.shape[0]

    # Angle step in radians from the model's actual view spacing.
    angle_step = float(np.median(np.abs(np.diff(angles))))
    if angle_step <= 0:
        raise ValueError("Model angles must have nonzero spacing.")
    views_per_frame = int(round(angle_span_per_frame / angle_step))
    stride          = int(round(angle_stride / angle_step))

    if views_per_frame <= 0:
        raise ValueError("angle_span_per_frame must cover at least one view.")
    if stride <= 0:
        raise ValueError("angle_stride must cover at least one view.")
    if views_per_frame > num_views:
        raise ValueError("angle_span_per_frame cannot exceed the full scan.")

    sino_frames = []
    model_frames = []
    for start in range(0, num_views - views_per_frame + 1, stride):
        sl = slice(start, start + views_per_frame)
        sino_frames.append(sino[sl])
        model_frames.append(mj.copy_ct_model(model, new_angles=angles[sl]))
    return sino_frames, model_frames


# ---------------------------------------------------------------------------
# DCT-I temporal dejitter
# ---------------------------------------------------------------------------

def dejitter_4d_dct(
    recon_4d,
    period,
    harmonics=True,
    band_width=1,
    dtype=np.float32,
    chunk_size=None,
    verbose=True,
):
    """
    Remove periodic temporal jitter from a 4D reconstruction via DCT-I filtering.

    Parameters
    ----------
    recon_4d : ndarray, shape (time, x, y, z)
    period : float or int
        Main jitter period in frames (e.g. 6 for a 6-phase gating protocol).
    harmonics : bool or list of int
        True  → remove main period and all harmonics with period/h >= 2.
        False → remove only the main period.
        list  → explicit list of harmonic indices h to remove.
    band_width : int
        Number of DCT-I modes to zero on each side of the target mode.
        band_width=1 zeroes [k_center-1, k_center, k_center+1].
    dtype : np.dtype
        Working dtype (float32 reduces memory).
    chunk_size : int or None
        Process the last spatial axis in chunks of this size to reduce peak
        memory. None processes the whole axis in one pass.
    verbose : bool
        Print the modes being zeroed.

    Returns
    -------
    recon_dejittered : ndarray, same shape as recon_4d
    """
    recon_4d = np.asarray(recon_4d)
    N = recon_4d.shape[0]
    spatial_shape = recon_4d.shape[1:]

    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        max_h = int(np.floor(period / 2))
        harmonic_list = list(range(1, max_h + 1))
    else:
        harmonic_list = list(harmonics)

    periods_to_remove = [period / h for h in harmonic_list]

    if verbose:
        print("Input shape:", recon_4d.shape)
        print("Periods to remove:", periods_to_remove)

    Z = spatial_shape[-1]
    if chunk_size is None:
        chunk_size = Z

    recon_dejittered = np.empty((N,) + spatial_shape, dtype=dtype)
    for z0 in range(0, Z, chunk_size):
        z1 = min(z0 + chunk_size, Z)
        block = np.asarray(recon_4d[..., z0:z1], dtype=dtype)
        C = dct(block, type=1, norm="ortho", axis=0)
        for p in periods_to_remove:
            k_center = 2 * (N - 1) / p
            k0 = int(round(k_center))
            lo = max(0, k0 - band_width)
            hi = min(C.shape[0], k0 + band_width + 1)
            if lo < hi:
                C[lo:hi, ...] = 0
            if verbose and z0 == 0:
                actual_period = 2 * (N - 1) / k0 if k0 != 0 else np.inf
                print(
                    f"  Removed period {p:.3g}: "
                    f"k≈{k_center:.2f}, rounded k={k0}, "
                    f"actual period≈{actual_period:.3g}, "
                    f"zeroed k={lo}:{hi - 1}"
                )
        recon_dejittered[..., z0:z1] = idct(C, type=1, norm="ortho", axis=0).astype(dtype, copy=False)
        del block, C
    return recon_dejittered


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def normalize_prior_weights(prior_weight):
    """
    Convert a scalar or list prior weight into [forward_w, xyt_w, yzt_w, xzt_w].

    Scalar w → [1-w, w/3, w/3, w/3].
    List/tuple [w1, w2, w3] → [1-(w1+w2+w3), w1, w2, w3].
    """
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = [float(w) for w in prior_weight]
        if len(prior) != 3:
            raise ValueError("prior_weight list must have 3 entries [xyt, yzt, xzt].")
    else:
        w = float(prior_weight) / 3.0
        prior = [w, w, w]
    if any(w < 0 for w in prior) or sum(prior) > 1.0:
        raise ValueError("prior weights must be nonnegative and sum to at most 1.")
    return [1.0 - sum(prior)] + prior


# ---------------------------------------------------------------------------
# Device-pinned denoiser helpers
# ---------------------------------------------------------------------------
#
# IMPORTANT: each QGGMRFDenoiser must be pinned to exactly ONE GPU via
# configure_devices([device]). Without this, mbirjax auto-shards the denoiser
# across every visible GPU using a NamedSharding Mesh. Running 4 such denoisers
# concurrently then causes each thread's model to open its own 4-way NCCL
# clique simultaneously — producing an "Acquire clique ... may be stuck" deadlock.
# Cache key includes the device so each thread gets its own pinned instance.

def get_qggmrf_denoiser(shape, device):
    """Return a per-thread, per-device cached QGGMRFDenoiser pinned to one GPU."""
    cache = getattr(_THREAD_LOCAL, "denoiser_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.denoiser_cache = cache
    key = (shape, device)
    if key not in cache:
        denoiser = mj.QGGMRFDenoiser(shape)
        denoiser.configure_devices([device])
        cache[key] = denoiser
    return cache[key]


def estimate_sigma_per_hyperplane(x, device, sigma_noise_floor=1e-6):
    """
    Estimate one noise sigma per hyperplane slice.

    Parameters
    ----------
    x : ndarray, shape (num_hyperplanes, dim1, dim2)
    device : jax device
        The denoiser is pinned to this device.

    Returns
    -------
    sigma_list : ndarray, shape (num_hyperplanes,), dtype float32
    """
    denoiser = get_qggmrf_denoiser(x.shape[1:], device)
    sigma_list = np.empty(x.shape[0], dtype=np.float32)
    for i in range(x.shape[0]):
        sigma_use = denoiser.estimate_image_noise_std(x[i][:, ::4, ::4])
        if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
            sigma_use = 0.0
        sigma_list[i] = sigma_use
    return sigma_list


def qggmrf_hyperplane_denoise(x, sigma_list, device, sigma_noise_floor=1e-6):
    """
    Denoise a stack of hyperplane slices on a single JAX device.

    Parameters
    ----------
    x : ndarray, shape (num_hyperplanes, dim1, dim2)
    sigma_list : ndarray, shape (num_hyperplanes,)
    device : jax device

    Returns
    -------
    y : ndarray, same shape as x
    """
    y = np.empty_like(x)
    with jax.default_device(device):
        denoiser = get_qggmrf_denoiser(x.shape[1:], device)
        for i in range(x.shape[0]):
            sigma_use = sigma_list[i]
            if (not np.isfinite(sigma_use)) or (sigma_use <= sigma_noise_floor):
                y[i] = x[i]
            else:
                image_i = jax.device_put(jnp.asarray(x[i]), device)
                y_i, _ = denoiser.denoise(image=image_i, sigma_noise=sigma_use)
                y[i] = np.asarray(y_i)
    return y


def denoiser_wrapper(x, permute_vector, sigma_list, device):
    """
    Permute a 4D volume, denoise the resulting hyperplane stack, then unpermute.

    Parameters
    ----------
    x : ndarray, shape (nt, nx, ny, nz)
    permute_vector : tuple of int
        Permutation that puts the hyperplane axis first.
    sigma_list : ndarray
        Per-hyperplane noise sigmas (from estimate_sigma_per_hyperplane).
    device : jax device
        Device on which denoising runs.

    Returns
    -------
    y : ndarray, same shape as x
    """
    x_perm = np.transpose(x, permute_vector)
    y_perm = qggmrf_hyperplane_denoise(x_perm, sigma_list=sigma_list, device=device)
    inv_perm = np.argsort(permute_vector)
    return np.transpose(y_perm, inv_perm)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def gen_gif_and_save(recon, gif_path, vmin=0, vmax=0.06, x_slice=None, duration=0.15):
    """
    Generate a GIF of one x slice (YZ plane) of a 4D reconstruction stepping
    through time frames. The x slice shows the motion clearly.

    Parameters
    ----------
    recon : ndarray, shape (nt, nx, ny, nz)
    gif_path : str
        Output path for the saved GIF.
    vmin, vmax : float
        Colormap range for imshow.
    x_slice : int or None
        X index to display. Defaults to the middle slice.
    duration : float
        Duration per frame in seconds.
    """
    if x_slice is None:
        x_slice = recon.shape[1] // 2

    frames = []
    for t in range(recon.shape[0]):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(recon[t, x_slice, :, :], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f't={t}')
        ax.axis('off')
        fig.suptitle(f'x slice = {x_slice}, time frame = {t}', fontsize=14)
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)

    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"Saved GIF to: {gif_path}")
