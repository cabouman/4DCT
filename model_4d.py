"""
4D MACE reconstruction model.

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

# Prior-agent hyperplane orientations. The permutation moves the hyperplane
# axis first; recon axes are (t, x, y, z).
PRIOR_ORIENTATIONS = [
    ("XY-t", (3, 0, 1, 2)),  # nz hyperplanes of shape (nt, nx, ny)
    ("YZ-t", (1, 0, 2, 3)),  # nx hyperplanes of shape (nt, ny, nz)
    ("XZ-t", (2, 0, 1, 3)),  # ny hyperplanes of shape (nt, nx, nz)
]

TIMING_FIELDS = [
    "iteration",
    "agent_0_forward_sec",
    "agent_1_prior_xyt_sec",
    "agent_2_prior_yzt_sec",
    "agent_3_prior_xzt_sec",
    "iteration_total_sec",
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
            True  → agents run concurrently, one GPU each (requires ≥4 GPUs).
            False → the same agents run sequentially on a single device.
        init_save_dir : str or None
            Directory where the computed init_image is saved as init_image.npy.
        timing_log_path : str or None
            CSV file for per-iteration agent timing.
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
        if init_image is None:
            init_image = self._compute_init_image(agent_devices[0], init_save_dir)
        else:
            init_image = np.asarray(init_image)
            if verbose:
                print("[MACE] Using provided init_image.")

        sigma_lists = self._compute_sigma_lists(init_image, agent_devices)

        # ── MACE state (all on CPU / NumPy) ────────────────────────────────────
        W = [np.copy(init_image) for _ in range(4)]
        X = [np.copy(init_image) for _ in range(4)]

        # ── Timing log ─────────────────────────────────────────────────────────
        if timing_log_path is not None:
            timing_log_dir = os.path.dirname(timing_log_path)
            if timing_log_dir:
                os.makedirs(timing_log_dir, exist_ok=True)
            with open(timing_log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=TIMING_FIELDS).writeheader()

        # ── Main MACE loop ─────────────────────────────────────────────────────
        for itr in range(self.max_mace_itr):
            itr_t0 = time.time()
            if verbose:
                print(f"\n[MACE] ── Iteration {itr + 1}/{self.max_mace_itr} ──")

            # Snapshot W so all agents see a consistent state this iteration.
            W_snap = [np.copy(W[k]) for k in range(4)]
            agent_times = {}

            if parallel:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {
                        pool.submit(self._run_forward_agent, W_snap[0], X[0],
                                    agent_devices[0]): (0, "forward"),
                    }
                    for k, (name, perm) in enumerate(PRIOR_ORIENTATIONS, start=1):
                        fut = pool.submit(self._run_prior_agent, W_snap[k], agent_devices[k],
                                          perm, sigma_lists[k - 1], name)
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
                X[0], agent_times[0] = self._run_forward_agent(W_snap[0], X[0], agent_devices[0])
                for k, (name, perm) in enumerate(PRIOR_ORIENTATIONS, start=1):
                    X[k], agent_times[k] = self._run_prior_agent(
                        W_snap[k], agent_devices[k], perm, sigma_lists[k - 1], name)

            # ADMM consensus (CPU)
            z = sum(beta[k] * (2.0 * X[k] - W[k]) for k in range(4))
            for k in range(4):
                W[k] = W[k] + 2.0 * self.rho * (z - X[k])

            iteration_sec = time.time() - itr_t0
            timing_row = dict(zip(
                TIMING_FIELDS,
                [itr + 1] + [agent_times[k] for k in range(4)] + [iteration_sec],
            ))
            if timing_log_path is not None:
                with open(timing_log_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=TIMING_FIELDS).writerow(timing_row)
            if verbose:
                print(
                    f"[MACE] Timing: itr={itr + 1}, "
                    + ", ".join(f"agent{k}={agent_times[k]:.2f}s" for k in range(4))
                    + f", total={iteration_sec:.2f}s"
                )

        if verbose:
            print("\n[MACE] Reconstruction complete.")

        return sum(beta[k] * X[k] for k in range(4))

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

    def _compute_sigma_lists(self, init_image, agent_devices):
        """Per-hyperplane noise sigmas for each prior orientation (one-time, device-pinned)."""
        if self.verbose:
            print("[MACE] Precomputing sigma lists...")
        sigma_lists = [
            estimate_sigma_per_hyperplane(
                np.transpose(init_image, perm), device=agent_devices[k + 1]
            )
            for k, (_, perm) in enumerate(PRIOR_ORIENTATIONS)
        ]
        if self.verbose:
            counts = ", ".join(
                f"{name}={np.count_nonzero(s > 1e-6)}/{s.size}"
                for (name, _), s in zip(PRIOR_ORIENTATIONS, sigma_lists)
            )
            print(f"[MACE] Nonzero sigma counts: {counts}")
            print("[MACE] Sigma precomputation done.")
        return sigma_lists

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

    def _run_prior_agent(self, W_k, device, permute_vector, sigma_list, agent_name):
        """Prior agent: qGGMRF denoising of one hyperplane orientation."""
        agent_t0 = time.time()
        out = denoiser_wrapper(self._dejitter(W_k), permute_vector=permute_vector,
                               sigma_list=sigma_list, device=device)
        agent_sec = time.time() - agent_t0
        if self.verbose:
            print(f"[MACE]  Prior agent {agent_name} ran on {device} in {agent_sec:.2f} sec.")
        return out, agent_sec

    def _dejitter(self, x):
        """Apply the DCT-I temporal dejitter if enabled; otherwise return x unchanged."""
        if not self.dejitter:
            return x
        return dejitter_4d_dct(x, period=self.dejitter_period, harmonics=True,
                               band_width=1, dtype=np.float32,
                               verbose=bool(self.verbose))

    def _compute_init_image(self, device, init_save_dir):
        """Per-frame MBIR recon (15 iterations) used as the MACE initial image."""
        if self.verbose:
            print(f"[MACE] Computing initial MBIR recon on {device} (one frame at a time)...")
        t0 = time.time()
        init_image = np.stack([
            np.asarray(
                self.model_list[t].recon(
                    jax.device_put(jnp.asarray(self.sino_list[t]), device),
                    weights=jax.device_put(jnp.asarray(self.weights_list[t]), device),
                    max_iterations=15,
                    stop_threshold_change_pct=self.stop_threshold,
                )[0]
            )
            for t in range(self.nt)
        ])
        if init_save_dir is not None:
            os.makedirs(init_save_dir, exist_ok=True)
            np.save(os.path.join(init_save_dir, "init_image.npy"), init_image)
        if self.verbose:
            print(f"[MACE] Initialization done in {time.time() - t0:.2f} sec.")
        return init_image
