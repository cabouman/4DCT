"""
4D MACE reconstruction — model and supporting utilities.

This module is one self-contained functional block (intended for a later merge
into mbirjax): the MACE4DModel class plus the helpers it uses for time-frame
construction, DCT-I temporal dejitter, hyperplane denoising, and visualization.

MACE4DModel runs the 4-agent MACE algorithm (one cone-beam prox_map agent +
three qGGMRF prior agents across XY-t, YZ-t, XZ-t hyperplanes) behind a single
recon() entry point. One MACE loop serves both execution modes:

  parallel=True  — agents run concurrently in a ThreadPoolExecutor, one GPU
                   each (requires ≥4 GPUs). GPU pinning prevents NCCL deadlock.
  parallel=False — the same agents run sequentially on jax.devices()[0].
"""
from __future__ import annotations

import concurrent.futures
import csv
import os
import threading
import time
import warnings

import jax
import jax.numpy as jnp
import mbirjax as mj
import numpy as np
from scipy.fft import dct, idct

# MBIR iterations for the per-frame initialization recon.
_INIT_MBIR_ITERATIONS = 15

# Prior-agent hyperplane orientations. The permutation moves the hyperplane
# axis first; recon axes are (t, x, y, z).
_PRIOR_ORIENTATIONS = [
    ("XY-t", (3, 0, 1, 2)),  # nz hyperplanes of shape (nt, nx, ny)
    ("YZ-t", (1, 0, 2, 3)),  # nx hyperplanes of shape (nt, ny, nz)
    ("XZ-t", (2, 0, 1, 3)),  # ny hyperplanes of shape (nt, nx, nz)
]

_TIMING_FIELDS = [
    "iteration",
    "agent_0_forward_sec",
    "agent_1_prior_xyt_sec",
    "agent_2_prior_yzt_sec",
    "agent_3_prior_xzt_sec",
    "iteration_total_sec",
    "consensus_change_pct",
]


class MACE4DModel:
    """
    4D MACE CT reconstruction model.

    Parameters
    ----------
    sino_list : list of ndarray
        Per-time-frame sinograms, each shape (views_per_frame, det_rows, det_cols).
    model_list : list of mbirjax.ConeBeamModel
        Per-time-frame cone-beam models (one per frame, built via copy_ct_model).
    prior_weight : float or list of float
        Weight given to prior agents.
        Scalar w  → [1-w, w/3, w/3, w/3] across [forward, xyt, yzt, xzt].
        3-list    → [1-sum, w0, w1, w2] (override each prior independently).
    rho : float
        ADMM step size (Mann iteration parameter). Default 0.5.
    max_mace_itr : int
        Number of outer MACE iterations. Default 10.
    num_prox_iterations : int
        Max prox_map iterations per MACE step. Default 3.
    stop_threshold : float
        Convergence threshold passed to prox_map. Default 0.02.
    weight_type : str
        Sinogram weight type for mj.gen_weights. Default "transmission_root".
    sigma_p : float or None
        Proximal map sigma. None lets mbirjax choose automatically.
    dejitter : bool
        Apply DCT-I temporal dejitter inside each agent. Default True.
    dejitter_period : int
        Jitter period in frames (frames per gating cycle). Default 6.
    verbose : int
        0 = silent, 1 = normal progress, 2 = debug.
    """

    def __init__(
        self,
        sino_list,
        model_list,
        prior_weight=0.5,
        rho=0.5,
        max_mace_itr=10,
        num_prox_iterations=3,
        stop_threshold=0.02,
        weight_type="transmission_root",
        sigma_p=None,
        dejitter=True,
        dejitter_period=6,
        verbose=1,
    ):
        if len(sino_list) != len(model_list):
            raise ValueError("sino_list and model_list must have the same length.")

        self.sino_list = sino_list
        self.model_list = model_list
        self.nt = len(sino_list)
        self.rho = rho
        self.max_mace_itr = max_mace_itr
        self.num_prox_iterations = num_prox_iterations
        self.stop_threshold = stop_threshold
        self.sigma_p = sigma_p
        self.verbose = verbose
        self.dejitter = dejitter
        self.dejitter_period = dejitter_period
        self.beta = _normalize_prior_weights(prior_weight)

        if verbose:
            print(f"[MACE4D] Building weights for {self.nt} time frames...")
        self.weights_list = [
            mj.gen_weights(jnp.asarray(s), weight_type=weight_type)
            for s in sino_list
        ]
        if verbose:
            print("[MACE4D] Weights built.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recon(
        self,
        init_image=None,
        parallel=True,
        init_dir=None,
        log_dir=None,
        device_indices=None,
    ):
        """
        Run 4D MACE reconstruction.

        Parameters
        ----------
        init_image : ndarray or None
            Initial 4D image, shape (nt, nx, ny, nz). A wrong shape raises
            ValueError. If None, the initial image comes from init_dir (see
            below) or is recomputed.
        parallel : bool
            True  → agents run concurrently, one GPU each (requires ≥4 GPUs).
            False → the same agents run sequentially on a single device.
        init_dir : str or None
            Cache directory for the computed initial image (init_image.npy).
            If it holds an image of the correct shape, that image is used;
            otherwise the initialization is recomputed and saved there.
        log_dir : str or None
            Directory for log files (run_info.txt, timing_log.csv). Created if
            needed. None writes no log files.
        device_indices : list of int or None
            GPU indices for [forward, prior_xyt, prior_yzt, prior_xzt] in
            parallel mode. Default: [0, 1, 2, 3].

        Returns
        -------
        recon_4d : ndarray, shape (nt, nx, ny, nz)
        """
        nt = self.nt
        beta = self.beta
        verbose = self.verbose

        agent_devices = self._setup_devices(parallel, device_indices)
        if verbose:
            print(f"[MACE] Start 4D reconstruction with {nt} time frames.")

        # ── Initialization ─────────────────────────────────────────────────────
        if init_image is not None:
            init_image = self._validate_init_image(init_image)
            init_source = "provided by caller"
            if verbose:
                print("[MACE] Using provided init_image.")
        else:
            if init_dir is not None:
                init_image = self._load_cached_init(init_dir)
            if init_image is not None:
                init_source = f"cached ({os.path.join(init_dir, 'init_image.npy')})"
            else:
                init_image = self._compute_init_image(agent_devices[0], init_dir)
                init_source = (f"computed ({self.nt} frames, "
                               f"{_INIT_MBIR_ITERATIONS} MBIR iterations each)")

        # ── Global denoiser sigma (one value for all orientations) ─────────────
        global_sigma = self._estimate_global_sigma(init_image, agent_devices[1])
        if verbose:
            print(f"[MACE] Global denoiser sigma = {global_sigma:.6g}")

        # ── MACE state (all on CPU / NumPy) ────────────────────────────────────
        W = [np.copy(init_image) for _ in range(4)]
        X = [np.copy(init_image) for _ in range(4)]

        # ── Log files ──────────────────────────────────────────────────────────
        timing_log_path = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._write_run_info(log_dir, parallel, agent_devices, init_source, global_sigma)
            timing_log_path = os.path.join(log_dir, "timing_log.csv")
            with open(timing_log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writeheader()

        # ── Main MACE loop ─────────────────────────────────────────────────────
        # xbar is the consensus average sum(beta[k] X[k]); its relative change
        # per iteration is the convergence measure logged in timing_log.csv.
        xbar = init_image
        for itr in range(self.max_mace_itr):
            itr_t0 = time.time()
            if verbose:
                print(f"\n[MACE] ── Iteration {itr + 1}/{self.max_mace_itr} ──")

            # Agents only read W; W is not written until the consensus update
            # after every agent has finished, so no snapshot copy is needed.
            agent_times = {}

            if parallel:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {
                        pool.submit(self._run_forward_agent, W[0], X[0],
                                    agent_devices[0]): (0, "forward"),
                    }
                    for k, (name, perm) in enumerate(_PRIOR_ORIENTATIONS, start=1):
                        fut = pool.submit(self._run_prior_agent, W[k], agent_devices[k],
                                          perm, global_sigma, name)
                        futures[fut] = (k, f"prior {name}")
                    for fut in concurrent.futures.as_completed(futures):
                        agent_id, agent_name = futures[fut]
                        X[agent_id], agent_times[agent_id] = fut.result()
                        if verbose:
                            print(
                                f"[MACE]  Agent {agent_id} ({agent_name}) done "
                                f"at +{time.time() - itr_t0:.2f} sec."
                            )
            else:
                X[0], agent_times[0] = self._run_forward_agent(W[0], X[0], agent_devices[0])
                for k, (name, perm) in enumerate(_PRIOR_ORIENTATIONS, start=1):
                    X[k], agent_times[k] = self._run_prior_agent(
                        W[k], agent_devices[k], perm, global_sigma, name)

            # ADMM consensus (CPU)
            z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
            for k in range(4):
                W[k] = W[k] + 2.0 * self.rho * (z - X[k])

            xbar_prev, xbar = xbar, sum(beta[k] * X[k] for k in range(4))
            denom = np.linalg.norm(xbar_prev)
            change_pct = 100.0 * np.linalg.norm(xbar - xbar_prev) / denom if denom > 0 else np.inf

            iteration_sec = time.time() - itr_t0
            timing_row = dict(zip(
                _TIMING_FIELDS,
                [itr + 1] + [agent_times[k] for k in range(4)] + [iteration_sec, change_pct],
            ))
            if timing_log_path is not None:
                with open(timing_log_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writerow(timing_row)
            if verbose:
                print(
                    f"[MACE] Timing: itr={itr + 1}, "
                    + ", ".join(f"agent{k}={agent_times[k]:.2f}s" for k in range(4))
                    + f", total={iteration_sec:.2f}s, change={change_pct:.4f}%"
                )

        if verbose:
            print("\n[MACE] Reconstruction complete.")

        return xbar

    # ------------------------------------------------------------------
    # Agents and helpers
    # ------------------------------------------------------------------

    def _setup_devices(self, parallel, device_indices):
        """Pick one device per agent and pin every per-frame model to Agent 0's device."""
        if parallel:
            devices = jax.devices("gpu")
            if len(devices) < 4:
                raise RuntimeError(f"Need at least 4 GPUs, found {len(devices)}.")
            if device_indices is None:
                device_indices = [0, 1, 2, 3]
            agent_devices = [devices[i] for i in device_indices]
            if self.verbose:
                print(f"[MACE] Found {len(devices)} GPU(s): {devices}")
                print(
                    "[MACE] GPU assignment: "
                    + ", ".join(f"Agent{k}->GPU{idx}" for k, idx in enumerate(device_indices))
                )
        else:
            agent_devices = [jax.devices()[0]] * 4

        # SINGLE-GPU FIX: pin each per-frame ConeBeamModel to one device.
        # Without this, every model auto-shards across all GPUs on first call,
        # causing all 4 agents to open a 4-way NCCL clique simultaneously →
        # deadlock ("Acquire clique ... may be stuck").
        for t in range(self.nt):
            self.model_list[t].configure_devices([agent_devices[0]])
        if self.verbose:
            print(f"[MACE] Pinned all {self.nt} ConeBeamModel instances to {agent_devices[0]}.")
        return agent_devices

    def _estimate_global_sigma(self, init_image, device):
        """One global noise sigma for all denoising, estimated from the init image."""
        # Merge (nt, nx) so the estimator sees a 3D array; it subsamples internally.
        image_3d = init_image.reshape(-1, init_image.shape[2], init_image.shape[3])
        denoiser = _get_qggmrf_denoiser(image_3d.shape, device)
        return float(denoiser.estimate_image_noise_std(image_3d))

    def _run_forward_agent(self, W_k, X_prev, device):
        """Agent 0: cone-beam prox_map, one time frame at a time, on one device."""
        agent_t0 = time.time()
        out = np.stack([
            np.asarray(
                self.model_list[t].prox_map(
                    prox_input=jax.device_put(jnp.asarray(W_k[t]), device),
                    sinogram=jax.device_put(jnp.asarray(self.sino_list[t]), device),
                    sigma_prox=self.sigma_p,
                    weights=jax.device_put(jnp.asarray(self.weights_list[t]), device),
                    init_recon=jax.device_put(jnp.asarray(X_prev[t]), device),
                    max_iterations=self.num_prox_iterations,
                    stop_threshold_change_pct=self.stop_threshold,
                )[0]
            )
            for t in range(self.nt)
        ])
        out = self._dejitter(out)
        agent_sec = time.time() - agent_t0
        if self.verbose:
            print(f"[MACE]  Forward agent ran on {device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def _run_prior_agent(self, W_k, device, permute_vector, sigma, agent_name):
        """Prior agent: batched qGGMRF denoising of one hyperplane orientation."""
        agent_t0 = time.time()
        out = _denoiser_wrapper(self._dejitter(W_k), permute_vector=permute_vector,
                               sigma=sigma, device=device)
        agent_sec = time.time() - agent_t0
        if self.verbose:
            print(f"[MACE]  Prior agent {agent_name} ran on {device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def _dejitter(self, x):
        """Apply the DCT-I temporal dejitter if enabled; otherwise return x unchanged."""
        if not self.dejitter:
            return x
        return _dejitter_4d_dct(x, period=self.dejitter_period, harmonics=True,
                               band_width=1, dtype=np.float32,
                               verbose=bool(self.verbose))

    def _write_run_info(self, log_dir, parallel, agent_devices, init_source, global_sigma):
        """Write a human-readable summary of the run settings to run_info.txt."""
        try:
            from importlib.metadata import version
            mbirjax_version = version("mbirjax")
        except Exception:
            mbirjax_version = "unknown"
        if parallel:
            mode = "parallel, agents on " + ", ".join(str(d) for d in agent_devices)
        else:
            mode = f"serial on {agent_devices[0]}"
        lines = [
            "# MACE4DModel run settings",
            f"date                 = {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"mbirjax version      = {mbirjax_version}",
            f"time frames (nt)     = {self.nt}",
            f"frame shape          = {tuple(self._expected_init_shape()[1:])}",
            f"mode                 = {mode}",
            f"init source          = {init_source}",
            f"beta [fwd, xyt, yzt, xzt] = {[round(float(b), 4) for b in self.beta]}",
            f"rho                  = {self.rho}",
            f"max_mace_itr         = {self.max_mace_itr}",
            f"num_prox_iterations  = {self.num_prox_iterations}",
            f"stop_threshold       = {self.stop_threshold}",
            f"sigma_p              = {'auto' if self.sigma_p is None else self.sigma_p}",
            f"denoiser sigma (global) = {global_sigma:.6g}",
            f"dejitter             = {self.dejitter}, period {self.dejitter_period}",
        ]
        with open(os.path.join(log_dir, "run_info.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def _expected_init_shape(self):
        """Shape the initial image must have: (nt,) + per-frame recon shape."""
        return (self.nt,) + tuple(self.model_list[0].get_params("recon_shape"))

    def _validate_init_image(self, init_image):
        """Return init_image as float32, or raise ValueError on a wrong shape."""
        init_image = np.asarray(init_image, dtype=np.float32)
        expected = self._expected_init_shape()
        if init_image.shape != expected:
            raise ValueError(
                f"init_image shape {init_image.shape} does not match expected {expected}."
            )
        return init_image

    def _load_cached_init(self, init_dir):
        """Load init_image.npy from init_dir if present and valid; else return None.

        A missing file is normal (first run) and silent. A file that cannot be
        loaded or has the wrong shape produces a warning.
        """
        path = os.path.join(init_dir, "init_image.npy")
        if not os.path.isfile(path):
            return None
        try:
            init_image = self._validate_init_image(np.load(path))
        except (ValueError, OSError) as e:
            warnings.warn(f"init_dir has an invalid initialization image ({e}); recomputing.")
            return None
        if self.verbose:
            print(f"[MACE] Using cached init from {path}.")
        return init_image

    def _compute_init_image(self, device, init_dir):
        """Per-frame MBIR recon used as the MACE initial image."""
        if self.verbose:
            print(f"[MACE] Computing initial MBIR recon on {device} (one frame at a time)...")
        t0 = time.time()
        init_image = np.stack([
            np.asarray(
                self.model_list[t].recon(
                    jax.device_put(jnp.asarray(self.sino_list[t]), device),
                    weights=jax.device_put(jnp.asarray(self.weights_list[t]), device),
                    max_iterations=_INIT_MBIR_ITERATIONS,
                    stop_threshold_change_pct=self.stop_threshold,
                )[0]
            )
            for t in range(self.nt)
        ])
        if init_dir is not None:
            os.makedirs(init_dir, exist_ok=True)
            np.save(os.path.join(init_dir, "init_image.npy"), init_image)
        if self.verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
        return init_image


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

def _dejitter_4d_dct(
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

def _normalize_prior_weights(prior_weight):
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

def _get_qggmrf_denoiser(shape, device):
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


# Denoiser iteration settings (match the old per-volume denoise() defaults).
_DENOISE_MAX_ITERATIONS = 15
_DENOISE_STOP_THRESHOLD_PCT = 0.2

# Working-set multiplier for the batch-size estimate: bytes used per volume
# during the jitted sweep, as a multiple of the volume size. Heuristic; refine
# by measurement on the target GPU.
_DENOISE_BUFFER_MULTIPLIER = 16


def _configure_denoiser(denoiser, sigma, image_for_stats):
    """Set the shared sigma and the regularization constants on the denoiser.

    Replicates the parameter setup that QGGMRFDenoiser.denoise() performs, so
    the jitted sweep can be called directly with shared constants.
    """
    denoiser.set_params(use_ror_mask=False, sigma_noise=float(sigma))
    verbose = denoiser.get_params('verbose')
    denoiser.set_params(verbose=0)
    denoiser.auto_set_regularization_params(image_for_stats)
    denoiser.set_params(verbose=verbose)
    # The sweep's progress callback converts its arguments with int()/float(),
    # which fails on the batched arrays a vmapped sweep passes it; silence it.
    denoiser._log_denoise_progress = lambda *args: None
    # Recompute the sweep constants for this configuration. The pixel partition
    # is random per generation, so it must be built once here and reused —
    # otherwise repeated calls run different VCD subset orders.
    denoiser._mace4d_constants = None
    _denoise_constants(denoiser)


def _denoise_constants(denoiser):
    """The constant arguments of the denoiser's jitted sweep, cached per configuration."""
    cached = getattr(denoiser, '_mace4d_constants', None)
    if cached is not None:
        return cached
    image_shape, granularity = denoiser.get_params(['recon_shape', 'granularity'])
    partition = mj.gen_set_of_pixel_partitions(image_shape, [granularity[0]],
                                               use_ror_mask=False)[0]
    fm_constant = 1.0 / (denoiser.get_params('sigma_y') ** 2.0)
    qggmrf_nbr_wts, sigma_x, p, q, T = denoiser.get_params(
        ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
    qggmrf_params = (mj.get_b_from_nbr_wts(qggmrf_nbr_wts), sigma_x, p, q, T)
    denoiser._mace4d_constants = (partition, fm_constant, qggmrf_params, image_shape)
    return denoiser._mace4d_constants


def _auto_batch_size(vol_shape, device):
    """Largest volume batch that fits in device memory; a small fixed batch on CPU."""
    stats = getattr(device, 'memory_stats', lambda: None)()
    if not stats:
        return 4
    free = stats.get('bytes_limit', 0) - stats.get('bytes_in_use', 0)
    vol_bytes = 4 * int(np.prod(vol_shape))
    return max(1, int(0.5 * free) // (_DENOISE_BUFFER_MULTIPLIER * vol_bytes))


def _batched_hyperplane_denoise(x, denoiser, device):
    """
    Denoise a stack of same-shaped 3D volumes with shared, preconfigured settings.

    One jax.vmap call runs the denoiser's single-device jitted sweep over a
    whole batch, so the GPU is filled instead of processing volumes one at a
    time (plan: Optimization Step 1). The volumes are independent, so the
    result equals per-volume denoising with the same constants.

    Parameters
    ----------
    x : ndarray, shape (num_volumes, d0, d1, d2)
    denoiser : QGGMRFDenoiser for shape (d0, d1, d2), after _configure_denoiser.
    device : jax device

    Returns
    -------
    y : ndarray, same shape as x
    """
    num_vols, vol_shape = x.shape[0], x.shape[1:]
    partition, fm_constant, qggmrf_params, image_shape = _denoise_constants(denoiser)
    stop_thresh = _DENOISE_STOP_THRESHOLD_PCT / 100.0

    def denoise_one(flat_vol):
        out, _, _, _ = denoiser._denoise_single_device(
            flat_vol, jnp.zeros_like(flat_vol), partition, fm_constant,
            qggmrf_params, image_shape, _DENOISE_MAX_ITERATIONS, stop_thresh, 0)
        return out

    batch = _auto_batch_size(vol_shape, device)
    flat = x.reshape(num_vols, -1, vol_shape[-1])
    y = np.empty_like(x)
    with jax.default_device(device):
        for b0 in range(0, num_vols, batch):
            b1 = min(b0 + batch, num_vols)
            block = jax.device_put(jnp.asarray(flat[b0:b1]), device)
            out = jax.vmap(denoise_one)(block)
            y[b0:b1] = np.asarray(out).reshape((b1 - b0,) + vol_shape)
    return y


def _denoiser_wrapper(x, permute_vector, sigma, device):
    """
    Permute a 4D volume so the hyperplane axis is first, batch-denoise the
    resulting stack of 3D volumes at the shared global sigma, then unpermute.

    Parameters
    ----------
    x : ndarray, shape (nt, nx, ny, nz)
    permute_vector : tuple of int
        Permutation that puts the hyperplane axis first.
    sigma : float
        Global noise sigma shared by every volume.
    device : jax device
        Device on which denoising runs.

    Returns
    -------
    y : ndarray, same shape as x
    """
    x_perm = np.ascontiguousarray(np.transpose(x, permute_vector))
    denoiser = _get_qggmrf_denoiser(x_perm.shape[1:], device)
    # Regularization statistics come from the whole stack (merged to 3D), so
    # every orientation sees the same voxel population.
    _configure_denoiser(denoiser, sigma, x_perm.reshape(-1, *x_perm.shape[2:]))
    y_perm = _batched_hyperplane_denoise(x_perm, denoiser, device)
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
    # Imported here so the compute path does not require visualization packages.
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

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
