"""
4D MACE reconstruction — model and supporting utilities.

This module is one self-contained functional block (intended for a later merge
into mbirjax): the MACE4DModel class plus the helpers it uses for time-frame
construction, DCT-I temporal dejitter, hyperplane denoising, and visualization.

MACE4DModel runs the MACE algorithm (one cone-beam prox_map per time frame +
three batched qGGMRF denoisers across the XY-t, YZ-t, XZ-t hyperplanes) behind
a single recon() entry point. Each iteration's work is a set of independent
tasks executed by one worker thread per device with a fixed least-loaded
assignment; a single device runs the same tasks inline.
"""
from __future__ import annotations

import concurrent.futures
import csv
import io
import logging
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
    "prox_total_sec",
    "denoise_total_sec",
    "makespan_sec",
    "iteration_total_sec",
    "consensus_change_pct",
]

_TASK_FIELDS = ["iteration", "kind", "index", "device", "start_sec", "end_sec"]

# Estimated cost of denoising one hyperplane, in units of one prox_map task.
# Measured on an H100 at smoke scale; only the relative size matters, and only
# for the static load-balancing assignment.
_DENOISE_COST_PER_PLANE = 0.015


class MACE4DModel:
    """
    4D MACE CT reconstruction model.

    Parameters
    ----------
    sino_list : list of ndarray
        Per-time-frame sinograms, each shape (views_per_frame, det_rows, det_cols).
    model_list : list of mbirjax.ConeBeamModel
        Per-time-frame cone-beam models (one per frame, built via copy_ct_model).
    mace_prior_weight : float or list of float
        Weight given to prior agents.
        Scalar w  → [1-w, w/3, w/3, w/3] across [forward, xyt, yzt, xzt].
        3-list    → [1-sum, w0, w1, w2] (override each prior independently).
    rho_mann : float
        Mann iteration step size (ADMM rho). Default 0.5.
    max_mace_itr : int
        Number of outer MACE iterations. Default 10.
    prox_num_iterations : int
        Max prox_map iterations per MACE step. Default 3.
    prox_stop_threshold : float
        Convergence threshold passed to prox_map. Default 0.02.
    weight_type : str
        Sinogram weight type for mj.gen_weights. Default "transmission_root".
    sigma_prox : float or None
        Proximal map sigma. None lets mbirjax choose automatically.
    dejitter : bool
        Apply DCT-I temporal dejitter inside each agent. Default True.
    frames_per_rotation : int
        Time frames per full 360 degree rotation. This is also the period
        of the jitter introduced by sinogram gating. Default 6.
    verbose : int
        0 = silent, 1 = normal progress, 2 = debug.
    """

    def __init__(
        self,
        sino_list,
        model_list,
        mace_prior_weight=0.5,
        rho_mann=0.5,
        max_mace_itr=10,
        prox_num_iterations=3,
        prox_stop_threshold=0.02,
        weight_type="transmission_root",
        sigma_prox=None,
        dejitter=True,
        frames_per_rotation=6,
        verbose=1,
    ):
        if len(sino_list) != len(model_list):
            raise ValueError("sino_list and model_list must have the same length.")

        self.sino_list = sino_list
        self.model_list = model_list
        self.nt = len(sino_list)
        self.rho_mann = rho_mann
        self.max_mace_itr = max_mace_itr
        self.prox_num_iterations = prox_num_iterations
        self.prox_stop_threshold = prox_stop_threshold
        self.sigma_prox = sigma_prox
        self.verbose = verbose
        self.dejitter = dejitter
        self.frames_per_rotation = frames_per_rotation
        self.beta = _normalize_prior_weights(mace_prior_weight)

        if verbose:
            print(f"[MACE4D] Building weights for {self.nt} time frames...")
        # Host copies; recon() places each frame's weights on its assigned device.
        self.weights_list = [
            np.asarray(mj.gen_weights(jnp.asarray(s), weight_type=weight_type))
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
        devices=None,
        init_dir=None,
        log_dir=None,
    ):
        """
        Run 4D MACE reconstruction.

        Each iteration is a set of independent tasks — one prox_map per time
        frame and one batched denoise per prior orientation — executed by one
        worker thread per device with a fixed least-loaded task assignment.

        Parameters
        ----------
        init_image : ndarray or None
            Initial 4D image, shape (nt, nx, ny, nz). A wrong shape raises
            ValueError. If None, the initial image comes from init_dir (see
            below) or is recomputed.
        devices : None, int, or list of jax devices
            None → all visible GPUs (the CPU when there are none).
            int n → the first n visible devices.  One device runs the tasks
            inline with no threads (the serial path).
        init_dir : str or None
            Cache directory for the computed initial image (init_image.npy).
            If it holds an image of the correct shape, that image is used;
            otherwise the initialization is recomputed and saved there.
        log_dir : str or None
            Directory for log files (run_info.txt, timing_log.csv,
            task_log.csv). Created if needed. None writes no log files.

        Returns
        -------
        recon_4d : ndarray, shape (nt, nx, ny, nz)
        """
        nt = self.nt
        beta = self.beta
        verbose = self.verbose

        devs = _resolve_devices(devices)
        self._assign_and_place(devs)
        if verbose:
            counts = [self._frame_device.count(d) for d in range(len(devs))]
            print(f"[MACE] {len(devs)} device(s); prox frames per device: {counts}; "
                  f"denoise on devices {self._orient_device}.")
            print(f"[MACE] Start 4D reconstruction with {nt} time frames.")

        # One single-thread executor per device: each device's tasks always run
        # on the same thread, which keeps the per-thread denoiser caches valid
        # and gives every model object exactly one owning thread.
        executors = ([concurrent.futures.ThreadPoolExecutor(max_workers=1) for _ in devs]
                     if len(devs) > 1 else None)
        # Denoisers reconfigure once per recon (sigma + regularization constants).
        self._recon_token = getattr(self, "_recon_token", 0) + 1
        try:
            # ── Initialization ─────────────────────────────────────────────────
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
                    init_image = self._compute_init_image(devs, executors, init_dir)
                    init_source = (f"computed ({self.nt} frames, "
                                   f"{_INIT_MBIR_ITERATIONS} MBIR iterations each)")

            # ── Global denoiser sigma (one value for all orientations) ─────────
            global_sigma = self._estimate_global_sigma(init_image, devs[0])
            if verbose:
                print(f"[MACE] Global denoiser sigma = {global_sigma:.6g}")

            # ── MACE state (all on CPU / NumPy) ────────────────────────────────
            W = [np.copy(init_image) for _ in range(4)]
            X = [np.copy(init_image) for _ in range(4)]
            # Reused every iteration by the consensus update below, so the
            # temp-heavy expression form (sum() over freshly allocated full-size
            # arrays) never runs -- see the in-place rewrite there.
            _consensus_scratch = np.empty_like(init_image)

            # ── Log files ──────────────────────────────────────────────────────
            timing_log_path = task_log_path = None
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                self._write_run_info(log_dir, devs, init_source, global_sigma)
                timing_log_path = os.path.join(log_dir, "timing_log.csv")
                with open(timing_log_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writeheader()
                task_log_path = os.path.join(log_dir, "task_log.csv")
                with open(task_log_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=_TASK_FIELDS).writeheader()

            # ── Main MACE loop ─────────────────────────────────────────────────
            # xbar is the consensus average sum(beta[k] X[k]); its relative
            # change per iteration is the convergence measure in timing_log.csv.
            xbar = init_image
            for itr in range(self.max_mace_itr):
                itr_t0 = time.time()
                if verbose:
                    print(f"\n[MACE] ── Iteration {itr + 1}/{self.max_mace_itr} ──")

                # Tasks only read W; W is not written until the consensus update
                # after the barrier, so no snapshot copy is needed.
                tasks = []
                for t in range(nt):
                    d = self._frame_device[t]
                    tasks.append((d, ("prox", t),
                                  lambda tt=t, dd=d: self._run_prox_task(tt, W[0][tt], X[0][tt], devs[dd])))
                for k in range(3):
                    d = self._orient_device[k]
                    perm = _PRIOR_ORIENTATIONS[k][1]
                    tasks.append((d, ("denoise", k),
                                  lambda kk=k, pp=perm, dd=d: self._run_denoise_task(
                                      W[kk + 1], pp, global_sigma, devs[dd])))
                results, task_rows = self._run_task_set(executors, tasks, itr_t0)

                # Gather in frame order, then dejitter the assembled stack.
                # X[0] keeps the dejittered stack — it feeds the next prox calls.
                X[0] = self._dejitter(np.stack([results[("prox", t)] for t in range(nt)]))
                for k in range(3):
                    X[k + 1] = results[("denoise", k)]

                # ADMM consensus (CPU). In-place: the equivalent expression
                # form (z = sum(beta[k]*(2*X[k]-W[k]) ...), W[k] = W[k] + ...)
                # allocates ~28 fresh full-size arrays per iteration and
                # measured 7.1x slower on the full-resolution volume (252.8s
                # vs 35.5s single-threaded on one CPU core; see run_notes.md
                # and parallelization_report.md). Same math, same order of
                # operations, verified to produce identical results.
                scratch = _consensus_scratch
                z = np.zeros_like(X[0])
                for k in range(4):
                    np.multiply(X[k], 2.0, out=scratch)
                    scratch -= W[k]
                    scratch *= beta[k]
                    z += scratch
                for k in range(4):
                    np.subtract(z, X[k], out=scratch)
                    scratch *= (2.0 * self.rho_mann)
                    W[k] += scratch

                xbar_prev = xbar
                xbar = np.zeros_like(X[0])
                for k in range(4):
                    np.multiply(X[k], beta[k], out=scratch)
                    xbar += scratch
                denom = np.linalg.norm(xbar_prev)
                change_pct = 100.0 * np.linalg.norm(xbar - xbar_prev) / denom if denom > 0 else np.inf

                iteration_sec = time.time() - itr_t0
                prox_total = sum(r[4] - r[3] for r in task_rows if r[0] == "prox")
                denoise_total = sum(r[4] - r[3] for r in task_rows if r[0] == "denoise")
                makespan = max(r[4] for r in task_rows)
                timing_row = dict(zip(_TIMING_FIELDS,
                                      [itr + 1, prox_total, denoise_total, makespan,
                                       iteration_sec, change_pct]))
                if timing_log_path is not None:
                    with open(timing_log_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writerow(timing_row)
                    with open(task_log_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=_TASK_FIELDS)
                        for kind, index, dev_idx, start, end in sorted(task_rows, key=lambda r: r[3]):
                            w.writerow(dict(zip(_TASK_FIELDS,
                                                [itr + 1, kind, index, dev_idx,
                                                 round(start, 3), round(end, 3)])))
                if verbose:
                    print(f"[MACE] Timing: itr={itr + 1}, prox={prox_total:.2f}s, "
                          f"denoise={denoise_total:.2f}s, makespan={makespan:.2f}s, "
                          f"total={iteration_sec:.2f}s, change={change_pct:.4f}%")
        finally:
            if executors is not None:
                for ex in executors:
                    ex.shutdown(wait=True)

        if verbose:
            print("\n[MACE] Reconstruction complete.")

        return xbar

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def _assign_and_place(self, devs):
        """Fix the task-to-device assignment, pin models, and place per-frame data.

        The assignment is computed once and reused for every iteration: each
        model object gets one owning thread, and each frame's sinogram and
        weights are uploaded to its device once and stay there.
        """
        recon_shape = tuple(self.model_list[0].get_params("recon_shape"))
        plane_counts = [recon_shape[2], recon_shape[0], recon_shape[1]]  # XY-t, YZ-t, XZ-t
        self._frame_device, self._orient_device = _assign_tasks(self.nt, plane_counts, len(devs))
        for t in range(self.nt):
            dev = devs[self._frame_device[t]]
            self.model_list[t].configure_devices([dev])
            _silence_model_logging(self.model_list[t], f"mace4d.frame{t}")
        self._sino_dev = [jax.device_put(np.asarray(self.sino_list[t]),
                                         devs[self._frame_device[t]]) for t in range(self.nt)]
        self._weights_dev = [jax.device_put(self.weights_list[t],
                                            devs[self._frame_device[t]]) for t in range(self.nt)]

    def _run_task_set(self, executors, tasks, t0):
        """Run tasks [(device_index, tag, fn)] and wait for all of them.

        Inline when executors is None (one device). Returns ({tag: result},
        [(kind, index, device_index, start, end)]) with times relative to t0.
        A failed task raises immediately, naming the task.
        """
        results = {}
        rows = []

        def run_one(fn):
            start = time.time() - t0
            out = fn()
            return out, start, time.time() - t0

        if executors is None:
            for dev_idx, tag, fn in tasks:
                out, start, end = run_one(fn)
                results[tag] = out
                rows.append((tag[0], tag[1], dev_idx, start, end))
            return results, rows

        futures = {}
        for dev_idx, tag, fn in tasks:
            futures[executors[dev_idx].submit(run_one, fn)] = (tag, dev_idx)
        for fut in concurrent.futures.as_completed(futures):
            tag, dev_idx = futures[fut]
            try:
                out, start, end = fut.result()
            except Exception as err:
                raise RuntimeError(f"task {tag} on device {dev_idx} failed") from err
            results[tag] = out
            rows.append((tag[0], tag[1], dev_idx, start, end))
        return results, rows

    def _run_prox_task(self, t, W0_t, X0_t, device):
        """One frame's proximal map on its assigned device."""
        return np.asarray(
            self.model_list[t].prox_map(
                prox_input=jax.device_put(W0_t, device),
                sinogram=self._sino_dev[t],
                sigma_prox=self.sigma_prox,
                weights=self._weights_dev[t],
                init_recon=jax.device_put(X0_t, device),
                max_iterations=self.prox_num_iterations,
                stop_threshold_change_pct=self.prox_stop_threshold,
                logfile_path=None,
                print_logs=False,
            )[0])

    def _run_denoise_task(self, W_k, permute_vector, sigma, device):
        """One orientation's batched qGGMRF denoise on its assigned device."""
        return _denoiser_wrapper(self._dejitter(W_k), permute_vector=permute_vector,
                                 sigma=sigma, device=device,
                                 config_token=self._recon_token)

    def _init_frame_task(self, t, device):
        """One frame's MBIR initialization recon on its assigned device."""
        return np.asarray(
            self.model_list[t].recon(
                self._sino_dev[t],
                weights=self._weights_dev[t],
                max_iterations=_INIT_MBIR_ITERATIONS,
                stop_threshold_change_pct=self.prox_stop_threshold,
                logfile_path=None,
                print_logs=False,
            )[0])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_global_sigma(self, init_image, device):
        """One global noise sigma for all denoising, estimated from the init image."""
        # Merge (nt, nx) so the estimator sees a 3D array; it subsamples internally.
        image_3d = init_image.reshape(-1, init_image.shape[2], init_image.shape[3])
        denoiser = mj.QGGMRFDenoiser(image_3d.shape)
        denoiser.configure_devices([device])
        return float(denoiser.estimate_image_noise_std(image_3d))

    def _dejitter(self, x):
        """Apply the DCT-I temporal dejitter if enabled; otherwise return x unchanged."""
        if not self.dejitter:
            return x
        return _dejitter_4d_dct(x, period=self.frames_per_rotation, harmonics=True,
                               band_width=1, dtype=np.float32,
                               verbose=bool(self.verbose))

    def _write_run_info(self, log_dir, devs, init_source, global_sigma):
        """Write a human-readable summary of the run settings to run_info.txt."""
        try:
            from importlib.metadata import version
            mbirjax_version = version("mbirjax")
        except Exception:
            mbirjax_version = "unknown"
        if len(devs) > 1:
            mode = f"task queue over {len(devs)} devices: " + ", ".join(str(d) for d in devs)
        else:
            mode = f"serial on {devs[0]}"
        lines = [
            "# MACE4DModel run settings",
            f"date                 = {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"mbirjax version      = {mbirjax_version}",
            f"time frames (nt)     = {self.nt}",
            f"frame shape          = {tuple(self._expected_init_shape()[1:])}",
            f"mode                 = {mode}",
            f"init source          = {init_source}",
            f"beta [fwd, xyt, yzt, xzt] = {[round(float(b), 4) for b in self.beta]}",
            f"rho_mann             = {self.rho_mann}",
            f"max_mace_itr         = {self.max_mace_itr}",
            f"prox_num_iterations  = {self.prox_num_iterations}",
            f"prox_stop_threshold  = {self.prox_stop_threshold}",
            f"sigma_prox           = {'auto' if self.sigma_prox is None else self.sigma_prox}",
            f"denoiser sigma (global) = {global_sigma:.6g}",
            f"dejitter             = {self.dejitter}, frames_per_rotation {self.frames_per_rotation}",
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

    def _compute_init_image(self, devs, executors, init_dir):
        """Per-frame MBIR recon used as the MACE initial image.

        Uses the same workers and frame-to-device assignment as the MACE loop,
        so compiled programs and resident data carry over.
        """
        if self.verbose:
            print(f"[MACE] Computing initial MBIR recon on {len(devs)} device(s)...")
        t0 = time.time()
        tasks = [(self._frame_device[t], ("init", t),
                  lambda tt=t: self._init_frame_task(tt, devs[self._frame_device[tt]]))
                 for t in range(self.nt)]
        results, _ = self._run_task_set(executors, tasks, t0)
        init_image = np.stack([results[("init", t)] for t in range(self.nt)])
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
# Device selection and task assignment
# ---------------------------------------------------------------------------

def _resolve_devices(devices):
    """Return the list of jax devices to use.

    None → all visible GPUs (the CPU when there are none).  int n → the first
    n visible devices.  A list of jax devices is used as given.
    """
    if devices is None:
        try:
            return jax.devices("gpu")
        except RuntimeError:
            return [jax.devices()[0]]
    if isinstance(devices, int):
        try:
            pool = jax.devices("gpu")
        except RuntimeError:
            pool = jax.devices()
        if not 1 <= devices <= len(pool):
            raise ValueError(f"devices={devices}, but {len(pool)} device(s) are visible.")
        return pool[:devices]
    return list(devices)


def _assign_tasks(num_frames, plane_counts, num_devices):
    """Fixed least-loaded-first assignment of tasks to devices.

    The denoise tasks (estimated cost proportional to their hyperplane count)
    are placed first, largest first; then each unit-cost prox task goes to the
    least-loaded device.

    Returns
    -------
    frame_device : list of int, device index for each frame's prox task
    orient_device : list of int, device index for each orientation's denoise
    """
    loads = [0.0] * num_devices
    orient_device = [0] * len(plane_counts)
    for k in sorted(range(len(plane_counts)), key=lambda k: -plane_counts[k]):
        d = loads.index(min(loads))
        orient_device[k] = d
        loads[d] += _DENOISE_COST_PER_PLANE * plane_counts[k]
    frame_device = [0] * num_frames
    for t in range(num_frames):
        d = loads.index(min(loads))
        frame_device[t] = d
        loads[d] += 1.0
    return frame_device, orient_device


def _silence_model_logging(model, name):
    """Give the model a private no-op logger.

    mbirjax's setup_logger rebuilds handlers on a logger that is shared by
    every model of one class, which races when tasks run concurrently. A
    private logger with a NullHandler and a no-op setup_logger removes the
    shared mutable state; progress still reaches the console via the MACE
    timing prints.
    """
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    model.logger = logger
    model.log_buffer = io.StringIO()
    model.setup_logger = lambda *args, **kwargs: None


# ---------------------------------------------------------------------------
# Time frame construction
# ---------------------------------------------------------------------------

def construct_time_frames(sino, model, frames_per_rotation=6, frame_overlap_factor=2.0):
    """
    Split a full sinogram into overlapping fixed-size time frames.

    The number of views per frame and the stride between frames are derived
    from the model's angle spacing, so they stay correct under view subsampling.

    Parameters
    ----------
    sino : ndarray, shape (num_views, det_rows, det_cols)
    model : mbirjax.ConeBeamModel
        Fully-built model for the full scan.
    frames_per_rotation : int
        Number of time frames per full 360 degree rotation. Default 6
        (one frame every 60 degrees).
    frame_overlap_factor : float
        Number of frames that share any given view. Each frame spans
        frame_overlap_factor * (360 / frames_per_rotation) degrees.
        Default 2.0 (each frame spans 120 degrees).

    Returns
    -------
    sino_frames : list of ndarray
        Trailing views that cannot form a full frame are discarded.
    model_frames : list of mbirjax.ConeBeamModel
        Per-frame models built via mj.copy_ct_model.
    """
    # Internal angular quantities in radians.
    angle_stride = 2.0 * np.pi / frames_per_rotation
    angle_span_per_frame = frame_overlap_factor * angle_stride

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
        raise ValueError("frame_overlap_factor gives a frame span smaller than one view.")
    if stride <= 0:
        raise ValueError("frames_per_rotation gives a stride smaller than one view.")
    if views_per_frame > num_views:
        raise ValueError("frame span cannot exceed the full scan.")

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

# Absolute cap on the denoise batch size, so a bad memory estimate cannot
# request an enormous compile.
_DENOISE_BATCH_CAP = 128

# Floor for the auto-estimated qGGMRF regularization scale (sigma_x). Guards
# against a batch whose statistics happen to come out at or near zero (e.g. a
# batch dominated by background-only hyperplanes), which would otherwise make
# the qGGMRF solver produce NaN for the whole batch.
_SIGMA_X_FLOOR = 1e-6


def _auto_set_regularization_params_full_batch(denoiser, image_for_stats):
    """Auto-set sigma_y/sigma_x/sigma_prox from the whole stats array.

    QGGMRFDenoiser.auto_set_regularization_params() (inherited from
    TomographyModel) internally calls subsample_views(..., num_real_views=
    sinogram_shape[0]). For a per-orientation batch of hyperplanes merged
    into one array with the hyperplane axis first, sinogram_shape[0] is nt
    (one hyperplane's time-frame count), not the batch size — so that call
    silently uses only the first hyperplane's statistics instead of the
    whole batch. This computes the same statistics directly on the full
    array instead, and floors sigma_x so it cannot come out at zero.
    """
    image_for_stats = np.asarray(image_for_stats)
    sino_indicator = denoiser._get_sino_indicator(image_for_stats)
    denoiser.auto_set_sigma_y(image_for_stats, sino_indicator)
    recon_std = denoiser._get_estimate_of_recon_std(image_for_stats, sino_indicator)
    if not np.isfinite(recon_std):
        recon_std = 0.0
    denoiser.auto_set_sigma_x(recon_std)
    denoiser.auto_set_sigma_prox(recon_std)
    sigma_x = denoiser.get_params('sigma_x')
    if not np.isfinite(sigma_x) or sigma_x < _SIGMA_X_FLOOR:
        denoiser.set_params(no_warning=True, sigma_x=np.float32(_SIGMA_X_FLOOR))


def _configure_denoiser(denoiser, sigma, image_for_stats):
    """Set the shared sigma and the regularization constants on the denoiser.

    Replicates the parameter setup that QGGMRFDenoiser.denoise() performs, so
    the jitted sweep can be called directly with shared constants.
    """
    denoiser.set_params(use_ror_mask=False, sigma_noise=float(sigma))
    verbose = denoiser.get_params('verbose')
    denoiser.set_params(verbose=0)
    _auto_set_regularization_params_full_batch(denoiser, image_for_stats)
    denoiser.set_params(verbose=verbose)
    # The sweep's progress callback converts its arguments with int()/float(),
    # which fails on the batched arrays a vmapped sweep passes it; silence it.
    denoiser._log_denoise_progress = lambda *args: None
    # Recompute the sweep constants for this configuration. The pixel partition
    # is random per generation, so it must be built once here and reused —
    # otherwise repeated calls run different VCD subset orders.
    denoiser._mace4d_constants = None
    _denoise_constants(denoiser)
    # New constants invalidate the cached batch size and compiled batch function.
    denoiser._mace4d_batch = None
    denoiser._mace4d_batched_fn = None


def _denoise_constants(denoiser):
    """The constant arguments of the denoiser's jitted sweep, cached per configuration."""
    cached = getattr(denoiser, '_mace4d_constants', None)
    if cached is not None:
        return cached
    image_shape, granularity = denoiser.get_params(['recon_shape', 'granularity'])
    # Keep at least ~64 pixels per VCD subset: with very small subsets the
    # qGGMRF line search can hit 0/0 in flat regions. At real volume sizes
    # this leaves the subset count unchanged.
    num_pixels = image_shape[0] * image_shape[1]
    num_subsets = max(1, min(granularity[0], num_pixels // 64))
    partition = mj.gen_set_of_pixel_partitions(image_shape, [num_subsets],
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

    # One fixed batch size and one compiled batch function per configuration.
    # The last block is padded to the fixed size so every call reuses the same
    # compiled program.
    if getattr(denoiser, "_mace4d_batch", None) is None:
        denoiser._mace4d_batch = min(_DENOISE_BATCH_CAP, num_vols,
                                     _auto_batch_size(vol_shape, device))
        denoiser._mace4d_batched_fn = jax.jit(jax.vmap(denoise_one))

    flat = x.reshape(num_vols, -1, vol_shape[-1])
    y = np.empty_like(x)
    b0 = 0
    with jax.default_device(device):
        while b0 < num_vols:
            batch = denoiser._mace4d_batch
            fn = denoiser._mace4d_batched_fn
            b1 = min(b0 + batch, num_vols)
            block = flat[b0:b1]
            if b1 - b0 < batch:
                pad = np.zeros((batch - (b1 - b0),) + block.shape[1:], dtype=block.dtype)
                block = np.concatenate([block, pad], axis=0)
            try:
                out = np.asarray(fn(jax.device_put(block, device)))
            except Exception as err:
                # Out of device memory: halve the batch and recompile once.
                if "RESOURCE_EXHAUSTED" in str(err) and denoiser._mace4d_batch > 1:
                    denoiser._mace4d_batch = max(1, denoiser._mace4d_batch // 2)
                    denoiser._mace4d_batched_fn = jax.jit(jax.vmap(denoise_one))
                    continue
                raise
            y[b0:b1] = out[: b1 - b0].reshape((b1 - b0,) + vol_shape)
            b0 = b1
    return y


def _denoiser_wrapper(x, permute_vector, sigma, device, config_token=None):
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
    config_token : hashable or None
        Configure the denoiser (sigma, regularization constants, partition)
        only when this token changes — once per recon. None reconfigures on
        every call.

    Returns
    -------
    y : ndarray, same shape as x
    """
    x_perm = np.ascontiguousarray(np.transpose(x, permute_vector))
    denoiser = _get_qggmrf_denoiser(x_perm.shape[1:], device)
    if config_token is None or getattr(denoiser, "_mace4d_token", None) != config_token:
        # Regularization statistics come from the whole stack (merged to 3D),
        # so every orientation sees the same voxel population.
        _configure_denoiser(denoiser, sigma, x_perm.reshape(-1, *x_perm.shape[2:]))
        denoiser._mace4d_token = config_token
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
