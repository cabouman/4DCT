# 4DCT Plan

## Current structure

```
4DCT/
├── plans/
│   ├── refactor_plan.md          ← this file
│   └── lilly_interface.md        ← parameter reference for Lilly operators
├── mace4d.py                     ← MACE4DModel class + all 4D helpers (one functional block)
├── recon_4d.py                   ← command-line reconstruction driver (all params as flags)
├── demo_4d.sh                    ← shell script: edit DATA_PATH and run
├── tests/                        ← fast CPU tests
├── output/                       ← gitignored
└── data/                         ← gitignored
```

## Next goal

Merge `mace4d.py` into mbirjax as one functional block, then port to mbirtorch.
Open questions for that merge: module name and location inside mbirjax, whether
`gen_gif_and_save` generalizes into a display utility (it must be useful beyond
4D to qualify), and docstring-format alignment with mbirjax conventions.


## Optimization

1) Denoiser optimization
§For the 4D dataset with size 98*260*260*728 (T, X, Y, Z), parallelization across four agents::
•Agent 1: forward prox for 98 3D volumes in serial (can be optimized in the future)
•Agent 2: 4D denoiser of the t-XY planes -- 728 3D volumes with size 260*260*98
•Agent 3: 4D denoiser of the t-YZ planes -- 260 3D volumes with size 260*728*98
•Agent 4: 4D denoiser of the t-XZ planes -- 260 3D volumes with size 260*728*98

A) The problem is that running the denoisers in series on 728 volumes each of size 260*260*98 is very slow.1
This is because a single denoiser call takes a long time and only uses a fraction of the GPU.
So we really want to run multiple volumes in parallel on the same GPU, but the number of parallel volumes will depend on the GPU's capabilities and the size of the volume.
This needs to be automated.

B) There is a load imbalance between the agents.
In particular, in this case Agent 2 is much slower because of the 728 volumes that must be denoised.
If we fix A, this might help a lot with B.

### Proposed solution (Claude, 2026-08-21)

Why it is slow: each denoiser call processes one small 3D volume — too little
work to fill an H100 — and pays fixed per-call overhead (dispatch, transfer,
Python loop). Agent 2 pays that overhead 728 times per MACE iteration.

Note on the imbalance: the three prior agents process the same total number of
voxels (728 × 260·260·98 = 260 × 260·728·98 = the full 4D array). The gap
between Agent 2 and Agents 3/4 is per-call overhead × call count (728 vs 260),
not compute. Fixing A should therefore largely fix B for the prior agents.

Design decision (Charlie, 2026-08-21): use a SINGLE GLOBAL sigma for all
denoising, not per-hyperplane sigmas. This makes batching a pure wrapper:
sigma is set once on the denoiser, so every sigma-derived constant inside the
jitted function is shared, and the batch call vmaps over the image arrays only.
No mbirjax change is needed. It also eliminates the 728+260+260 per-hyperplane
sigma estimates (one global estimate replaces them) and the zero-sigma
skip-identity branch. Note: results will differ from the per-hyperplane-sigma
baselines — this is an intentional algorithm change.

Step 1 — Batched-denoise wrapper in mace4d.py.
Stack B same-shaped hyperplane volumes, replicate denoise()'s prologue
(flatten, partition), and make one jax.vmap call over the single-device jitted
sweep (QGGMRFDenoiser._denoise_single_device). The volumes are independent —
the qGGMRF neighborhood never crosses the batch axis — so batched output must
equal serial output exactly at the same sigma; add a test for that equality.
The vmapped convergence loop runs all lanes until the slowest converges,
which is the intended batch semantics. Caveat: the wrapper calls a private
mbirjax method, so it couples to mbirjax internals; at merge time the batched
path moves inside QGGMRFDenoiser properly.

Step 2 — Automate the batch size B.
B = floor(usable_device_memory / (k × volume_bytes)), capped by volumes
remaining, where k is the number of VCD-internal buffers. Measure k once
empirically (allocate a batch, read peak memory) rather than deriving it.
No user parameter; correct across GPU types and volume sizes.

Step 3 — Re-measure, then decide about the forward agent.
After batching, the residual imbalance is forward (98 sequential prox_maps)
vs priors. If timing shows forward dominates: the 98 frame prox_maps are
independent, so make the forward agent a frame queue — prior agents that
finish early take frames from the queue on their own GPUs (needs per-GPU
pinned model copies; copy_ct_model makes these cheap). Do not build until
Step 1 timing data justifies it.

Coordination: with the single-sigma design, no mbirjax change is needed now.
When mace4d merges into mbirjax, the batched path should move inside
QGGMRFDenoiser as a supported interface; align that with the mbirtorch port
(Greg is active there).

