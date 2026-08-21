"""
4D MACE reconstruction model.

MACE4DModel wraps the 4-agent MACE algorithm (one cone-beam prox_map agent +
three qGGMRF prior agents across XY-t, YZ-t, XZ-t hyperplanes) in a class
with a single recon() entry point. Two execution modes are available:

  parallel=True  — ThreadPoolExecutor(4), one agent per GPU. Requires ≥4 GPUs.
                   All GPU-pinning and NCCL-safety logic lives here.
  parallel=False — Sequential agents on jax.devices()[0].

The multi-GPU implementation is preserved verbatim from
4DMACE_multi_threads/utils_multi_threads.py; only the wrapping is new.
"""
from __future__ import annotations

import concurrent.futures
import csv
import os
import time

import jax
import jax.numpy as jnp
import mbirjax as mj
import numpy as np

from utils import (
    denoiser_wrapper,
    dejitter_4d_dct,
    estimate_sigma_per_hyperplane,
    normalize_prior_weights,
)


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
        self.beta = normalize_prior_weights(prior_weight)

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
        init_save_dir=None,
        timing_log_path=None,
        device_indices=None,
    ):
        """
        Run 4D MACE reconstruction.

        Parameters
        ----------
        init_image : ndarray or None
            Initial 4D image, shape (nt, nx, ny, nz). If None, a per-frame
            MBIR recon is computed automatically and saved to init_save_dir.
        parallel : bool
            True  → 4-GPU ThreadPoolExecutor (requires ≥4 GPUs).
            False → sequential agents on a single device.
        init_save_dir : str or None
            Directory where the computed init_image is saved as init_image.npy.
        timing_log_path : str or None
            CSV file for per-iteration agent timing (parallel mode only).
        device_indices : list of int or None
            GPU indices for [forward, prior_xyt, prior_yzt, prior_xzt].
            Default: [0, 1, 2, 3].

        Returns
        -------
        recon_4d : ndarray, shape (nt, nx, ny, nz)
        """
        if parallel:
            return self._recon_parallel(
                init_image=init_image,
                init_save_dir=init_save_dir,
                timing_log_path=timing_log_path,
                device_indices=device_indices,
            )
        else:
            return self._recon_serial(
                init_image=init_image,
                init_save_dir=init_save_dir,
            )

    # ------------------------------------------------------------------
    # Multi-GPU implementation (ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _recon_parallel(self, init_image, init_save_dir, timing_log_path, device_indices):
        nt = self.nt
        sino_list = self.sino_list
        weights_list = self.weights_list
        models = self.model_list
        beta = self.beta
        sigma_p = self.sigma_p
        verbose = self.verbose

        # ── GPU discovery ──────────────────────────────────────────────────────
        devices = jax.devices("gpu")
        n_gpu = len(devices)
        if n_gpu == 0:
            raise RuntimeError("No GPU devices found by JAX.")
        if n_gpu < 4:
            raise RuntimeError(f"Need at least 4 GPUs, found {n_gpu}.")

        if device_indices is None:
            device_indices = [0, 1, 2, 3]

        if verbose:
            print(f"[MACE] Found {n_gpu} GPU(s): {devices}")
            print(
                "[MACE] GPU assignment: "
                + ", ".join(f"Agent{k}->GPU{idx}" for k, idx in enumerate(device_indices))
            )
            print(f"[MACE] Start 4D reconstruction with {nt} time frames.")

        # ── SINGLE-GPU FIX: pin each per-frame ConeBeamModel to Agent 0's GPU ──
        # Without this, every model auto-shards across all GPUs on first call,
        # causing all 4 agents to open a 4-way NCCL clique simultaneously →
        # deadlock ("Acquire clique ... may be stuck").
        forward_device = devices[device_indices[0]]
        for t in range(nt):
            models[t].configure_devices([forward_device])
        if verbose:
            print(f"[MACE] Pinned all {nt} ConeBeamModel instances to {forward_device}.")

        # ── Initialization ─────────────────────────────────────────────────────
        init_device = devices[device_indices[0]]
        if init_image is None:
            if verbose:
                print(f"[MACE] Computing initial MBIR recon on GPU {device_indices[0]} (serial)...")
            t0 = time.time()
            init_image = np.stack([
                np.asarray(
                    models[t].recon(
                        jax.device_put(jnp.asarray(sino_list[t]), init_device),
                        weights=jax.device_put(jnp.asarray(weights_list[t]), init_device),
                        max_iterations=15,
                        stop_threshold_change_pct=self.stop_threshold,
                    )[0]
                )
                for t in range(nt)
            ])
            if init_save_dir is not None:
                os.makedirs(init_save_dir, exist_ok=True)
                np.save(os.path.join(init_save_dir, "init_image.npy"), init_image)
            if verbose:
                print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
        else:
            init_image = np.asarray(init_image)
            if verbose:
                print("[MACE] Using provided init_image.")

        # ── Sigma precomputation (one-time, device-pinned) ────────────────────
        if verbose:
            print("[MACE] Precomputing sigma lists...")
        sigma_xyt = estimate_sigma_per_hyperplane(
            np.transpose(init_image, (3, 0, 1, 2)), device=devices[device_indices[1]]
        )
        sigma_yzt = estimate_sigma_per_hyperplane(
            np.transpose(init_image, (1, 0, 2, 3)), device=devices[device_indices[2]]
        )
        sigma_xzt = estimate_sigma_per_hyperplane(
            np.transpose(init_image, (2, 0, 1, 3)), device=devices[device_indices[3]]
        )
        if verbose:
            print(
                "[MACE] Nonzero sigma counts: "
                f"XY-t={np.count_nonzero(sigma_xyt > 1e-6)}/{sigma_xyt.size}, "
                f"YZ-t={np.count_nonzero(sigma_yzt > 1e-6)}/{sigma_yzt.size}, "
                f"XZ-t={np.count_nonzero(sigma_xzt > 1e-6)}/{sigma_xzt.size}"
            )
            print("[MACE] Sigma precomputation done.")

        # ── MACE state (all on CPU / NumPy) ───────────────────────────────────
        W = [np.copy(init_image) for _ in range(4)]
        X = [np.copy(init_image) for _ in range(4)]

        # ── Timing log ────────────────────────────────────────────────────────
        if timing_log_path is not None:
            timing_log_dir = os.path.dirname(timing_log_path)
            if timing_log_dir:
                os.makedirs(timing_log_dir, exist_ok=True)
            with open(timing_log_path, "w", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=[
                        "iteration",
                        "agent_0_forward_sec",
                        "agent_1_prior_xyt_sec",
                        "agent_2_prior_yzt_sec",
                        "agent_3_prior_xzt_sec",
                        "iteration_total_sec",
                    ],
                ).writeheader()

        # ── Agent closures ─────────────────────────────────────────────────────
        # Closures capture: devices, device_indices, sino_list, weights_list,
        # models, sigma_*, sigma_p, nt, self.*. They are submitted to the
        # ThreadPoolExecutor below; each runs on its own OS thread.

        def run_forward_agent(W_k, X_prev, device_index):
            """Agent 0: cone-beam prox_map, serial over time frames, one GPU."""
            device = devices[device_index]
            agent_t0 = time.time()
            out = np.stack([
                np.asarray(
                    models[t].prox_map(
                        prox_input=jax.device_put(jnp.asarray(W_k[t]), device),
                        sinogram=jax.device_put(jnp.asarray(sino_list[t]), device),
                        sigma_prox=sigma_p,
                        weights=jax.device_put(jnp.asarray(weights_list[t]), device),
                        init_recon=jax.device_put(jnp.asarray(X_prev[t]), device),
                        max_iterations=self.num_prox_iterations,
                        stop_threshold_change_pct=self.stop_threshold,
                    )[0]
                )
                for t in range(nt)
            ])
            if self.dejitter:
                out = dejitter_4d_dct(
                    out, period=self.dejitter_period,
                    harmonics=True, band_width=1,
                    chunk_size=None, dtype=np.float32, verbose=bool(verbose),
                )
            agent_sec = time.time() - agent_t0
            if verbose:
                print(f"[MACE]  Agent 0 ran on {device} in {agent_sec:.2f} sec.")
            return out, agent_sec

        def run_prior_agent_1(W_k, device_index):
            """Agent 1: qGGMRF XY-t hyperplanes (fixed z slabs)."""
            device = devices[device_index]
            agent_t0 = time.time()
            if self.dejitter:
                W_k = dejitter_4d_dct(
                    W_k, period=self.dejitter_period,
                    harmonics=True, band_width=1,
                    chunk_size=None, dtype=np.float32, verbose=bool(verbose),
                )
            out = denoiser_wrapper(W_k, permute_vector=(3, 0, 1, 2), sigma_list=sigma_xyt, device=device)
            agent_sec = time.time() - agent_t0
            if verbose:
                print(f"[MACE]  Agent 1 ran on {device} in {agent_sec:.2f} sec.")
            return out, agent_sec

        def run_prior_agent_2(W_k, device_index):
            """Agent 2: qGGMRF YZ-t hyperplanes (fixed row slabs)."""
            device = devices[device_index]
            agent_t0 = time.time()
            if self.dejitter:
                W_k = dejitter_4d_dct(
                    W_k, period=self.dejitter_period,
                    harmonics=True, band_width=1,
                    chunk_size=None, dtype=np.float32, verbose=bool(verbose),
                )
            out = denoiser_wrapper(W_k, permute_vector=(1, 0, 2, 3), sigma_list=sigma_yzt, device=device)
            agent_sec = time.time() - agent_t0
            if verbose:
                print(f"[MACE]  Agent 2 ran on {device} in {agent_sec:.2f} sec.")
            return out, agent_sec

        def run_prior_agent_3(W_k, device_index):
            """Agent 3: qGGMRF XZ-t hyperplanes (fixed col slabs)."""
            device = devices[device_index]
            agent_t0 = time.time()
            if self.dejitter:
                W_k = dejitter_4d_dct(
                    W_k, period=self.dejitter_period,
                    harmonics=True, band_width=1,
                    chunk_size=None, dtype=np.float32, verbose=bool(verbose),
                )
            out = denoiser_wrapper(W_k, permute_vector=(2, 0, 1, 3), sigma_list=sigma_xzt, device=device)
            agent_sec = time.time() - agent_t0
            if verbose:
                print(f"[MACE]  Agent 3 ran on {device} in {agent_sec:.2f} sec.")
            return out, agent_sec

        # ── Main MACE loop ─────────────────────────────────────────────────────
        for itr in range(self.max_mace_itr):
            itr_t0 = time.time()
            if verbose:
                print(f"\n[MACE] ── Iteration {itr + 1}/{self.max_mace_itr} ──")

            # Snapshot W so all agents see a consistent state this iteration.
            W_snap = [np.copy(W[k]) for k in range(4)]
            agent_times = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(run_forward_agent, W_snap[0], X[0], device_indices[0]): (0, "forward"),
                    pool.submit(run_prior_agent_1, W_snap[1], device_indices[1]): (1, "prior XY-t"),
                    pool.submit(run_prior_agent_2, W_snap[2], device_indices[2]): (2, "prior YZ-t"),
                    pool.submit(run_prior_agent_3, W_snap[3], device_indices[3]): (3, "prior XZ-t"),
                }
                for fut in concurrent.futures.as_completed(futures):
                    agent_id, agent_name = futures[fut]
                    done_t0 = time.time()
                    X[agent_id], agent_times[agent_id] = fut.result()
                    if verbose:
                        print(
                            f"[MACE]  Agent {agent_id} ({agent_name}) done "
                            f"at +{done_t0 - itr_t0:.2f} sec."
                        )

            if verbose:
                print("[MACE]  All agents done. Running consensus update...")

            # ADMM consensus (CPU)
            z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
            for k in range(4):
                W[k] = W[k] + 2.0 * self.rho * (z - X[k])

            iteration_sec = time.time() - itr_t0
            timing_row = {
                "iteration": itr + 1,
                "agent_0_forward_sec": agent_times[0],
                "agent_1_prior_xyt_sec": agent_times[1],
                "agent_2_prior_yzt_sec": agent_times[2],
                "agent_3_prior_xzt_sec": agent_times[3],
                "iteration_total_sec": iteration_sec,
            }

            if timing_log_path is not None:
                with open(timing_log_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=timing_row.keys())
                    writer.writerow(timing_row)

            if verbose:
                print(
                    f"[MACE] Timing: itr={itr + 1}, "
                    f"agent0={agent_times[0]:.2f}s, "
                    f"agent1={agent_times[1]:.2f}s, "
                    f"agent2={agent_times[2]:.2f}s, "
                    f"agent3={agent_times[3]:.2f}s, "
                    f"total={iteration_sec:.2f}s"
                )

        if verbose:
            print("\n[MACE] Reconstruction complete.")

        return sum(beta[k] * X[k] for k in range(4))

    # ------------------------------------------------------------------
    # Serial implementation (single device)
    # ------------------------------------------------------------------

    def _recon_serial(self, init_image, init_save_dir):
        nt = self.nt
        sino_list = self.sino_list
        weights_list = self.weights_list
        models = self.model_list
        beta = self.beta
        sigma_p = self.sigma_p
        verbose = self.verbose

        device = jax.devices()[0]

        if verbose:
            print(f"[MACE] Start serial 4D reconstruction with {nt} time frames on {device}.")

        if init_image is None:
            if verbose:
                print("[MACE] Computing initial MBIR recon (serial)...")
            t0 = time.time()
            init_image = np.stack([
                np.asarray(
                    models[t].recon(
                        jnp.asarray(sino_list[t]),
                        weights=jnp.asarray(weights_list[t]),
                        max_iterations=20,
                        stop_threshold_change_pct=self.stop_threshold,
                    )[0]
                )
                for t in range(nt)
            ])
            if init_save_dir is not None:
                os.makedirs(init_save_dir, exist_ok=True)
                np.save(os.path.join(init_save_dir, "init_image.npy"), init_image)
            if verbose:
                print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
        else:
            init_image = np.asarray(init_image)
            if verbose:
                print("[MACE] Using provided init_image.")

        if verbose:
            print("[MACE] Precomputing sigma lists...")
        sigma_xyt = estimate_sigma_per_hyperplane(np.transpose(init_image, (3, 0, 1, 2)), device=device)
        sigma_yzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (1, 0, 2, 3)), device=device)
        sigma_xzt = estimate_sigma_per_hyperplane(np.transpose(init_image, (2, 0, 1, 3)), device=device)
        if verbose:
            print("[MACE] Sigma precomputation done.")

        W = [np.copy(init_image) for _ in range(4)]
        X = [np.copy(init_image) for _ in range(4)]

        for itr in range(self.max_mace_itr):
            itr_t0 = time.time()
            if verbose:
                print(f"[MACE] Iteration {itr + 1}/{self.max_mace_itr}")

            # Forward agent
            X[0] = np.stack([
                np.asarray(
                    models[t].prox_map(
                        prox_input=jnp.asarray(W[0][t]),
                        sinogram=jnp.asarray(sino_list[t]),
                        sigma_prox=sigma_p,
                        weights=jnp.asarray(weights_list[t]),
                        init_recon=jnp.asarray(X[0][t]),
                        max_iterations=self.num_prox_iterations,
                        stop_threshold_change_pct=self.stop_threshold,
                    )[0]
                )
                for t in range(nt)
            ])
            if self.dejitter:
                X[0] = dejitter_4d_dct(
                    X[0], period=self.dejitter_period,
                    harmonics=True, band_width=1, dtype=np.float32, verbose=bool(verbose),
                )

            # Prior agents
            W1 = dejitter_4d_dct(W[1], period=self.dejitter_period, harmonics=True, band_width=1, dtype=np.float32, verbose=bool(verbose)) if self.dejitter else W[1]
            W2 = dejitter_4d_dct(W[2], period=self.dejitter_period, harmonics=True, band_width=1, dtype=np.float32, verbose=bool(verbose)) if self.dejitter else W[2]
            W3 = dejitter_4d_dct(W[3], period=self.dejitter_period, harmonics=True, band_width=1, dtype=np.float32, verbose=bool(verbose)) if self.dejitter else W[3]

            X[1] = denoiser_wrapper(W1, permute_vector=(3, 0, 1, 2), sigma_list=sigma_xyt, device=device)
            X[2] = denoiser_wrapper(W2, permute_vector=(1, 0, 2, 3), sigma_list=sigma_yzt, device=device)
            X[3] = denoiser_wrapper(W3, permute_vector=(2, 0, 1, 3), sigma_list=sigma_xzt, device=device)

            # ADMM consensus
            z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
            for k in range(4):
                W[k] = W[k] + 2.0 * self.rho * (z - X[k])

            if verbose:
                print(f"[MACE] Iteration {itr + 1} done in {time.time() - itr_t0:.2f} sec.")

        if verbose:
            print("[MACE] Reconstruction complete.")

        return sum(beta[k] * X[k] for k in range(4))
