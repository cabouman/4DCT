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

### Done: batched denoisers with a single global sigma (2026-08-21)

Each denoiser call now processes a whole batch of hyperplane volumes in one
vmapped call, using one global sigma estimated once from the initialization
image.  Measured on the smoke configuration (25 frames, 8x4 downsampling,
4 H100s): prior agents went from 132 / 34 / 34 seconds per iteration to
2.5 / 1.2 / 1.2 seconds, and one MACE iteration went from 132 seconds to
23 seconds.  The batched output equals the serial output exactly at the same
sigma; the test suite checks this.  The forward model now sets the iteration
time.  At merge time the batched path should move inside QGGMRFDenoiser as a
supported interface; align that with the mbirtorch port (Greg is active there).

### Next: one GPU task queue for all agent work (planned 2026-08-21; revised after Opus panel review)

The forward model now limits the iteration time.  Its 25 proximal maps run in
series on one GPU while the other three GPUs sit mostly idle.  The proximal
maps are independent of each other, so they can run on different GPUs at the
same time.

**Design.**  Each MACE iteration produces 28 independent tasks: one proximal
map per time frame (25 tasks) and one batched denoise per prior orientation
(3 tasks).  N worker threads execute the tasks, one worker per GPU.  Every
task runs entirely on one GPU; no task is split across GPUs.  The old fixed
roles (one forward agent, three prior agents) disappear.  The workers are N
single-thread executors, one per GPU — not one shared pool — so each GPU's
tasks always run on the same thread.  That is what keeps the per-thread
denoiser caches valid and guarantees each model object is touched by exactly
one thread.

```
Task list (assignment computed once, reused every iteration; N = 4 shown)

  1. Sort the 28 tasks by estimated cost (the 3 denoise tasks first,
     then the 25 equal prox tasks).
  2. Assign each task to the currently least-loaded GPU.
  3. Freeze the assignment for the whole recon.

  GPU 0 worker: denoise XY-t, then prox frames ...   (~7.2 s)
  GPU 1 worker: denoise YZ-t, then prox frames ...   (~7.2 s)
  GPU 2 worker: denoise XZ-t, then prox frames ...   (~7.2 s)
  GPU 3 worker: prox frames only                     (~7.2 s)
       |
       v
  barrier: wait on every task's future (a failure raises with its task id)
       -> gather frames in frame order -> dejitter forward stack
       -> consensus update
```

**Why the assignment is fixed (sticky).**  Two reasons, both verified against
the mbirjax code during review.  First, thread ownership: each frame's
cone-beam model object carries mutable state (device placement, prox data,
log buffer), so it must only ever be used by one thread; a fixed map gives
each model one owner.  Second, data residency: with a fixed map, each frame's
sinogram and weights are uploaded to its GPU once at setup and stay there.
The current code re-uploads them every iteration; at full scale that is tens
of GB per iteration, funneled through GPU 0.  NOTE: an earlier version of
this plan justified stickiness by compilation cost.  That was wrong — mbirjax
compiles its projectors at module level and shares the compiled program
across all models with the same geometry on one device, so a dynamic queue
would cost about N compilations, not 25 x N.

**Assignment rule.**  Least-loaded-first (as in the diagram), not plain round
robin.  Round robin appends the three denoise tasks to GPUs that already hold
six prox tasks each, which makes the slowest GPU the limit: 7.9 s instead of
7.2 s at smoke scale, and about 22% worse than optimal at 8 GPUs.  Sorting by
cost and assigning to the least-loaded GPU costs a few lines and removes the
gap.  If profiling later shows imbalance — measure it as
(makespan − mean per-GPU busy time) / makespan, act above about 10% — the
response is to re-estimate the task costs and recompute the fixed map once,
not to make the queue dynamic.  A later refinement that makes any assignment
near-optimal: split each denoise into blocks of hyperplanes so all tasks are
close to uniform (every block must reuse the same pixel partition).

**Known crash risk to mitigate: the shared mbirjax logger.**  All models of
one class share a single process-global logger object, and every prox_map
call tears down and rebuilds its file handlers.  With concurrent prox calls,
one thread can close a handler while another thread writes to it.
Mitigation now: pass logfile_path=None and print_logs=False from worker
tasks, and hold one lock around model initialization.  Proper fix at merge
time: per-instance loggers in mbirjax.

**Data placement.**  At setup, place each frame's sinogram and weights
directly on that frame's assigned GPU (device_put on the numpy array — the
current jnp.asarray-then-device_put pattern stages everything through GPU 0
first).  Stop building the weights on the default device in __init__.

**Denoiser batching fixes (found in review).**  The vmapped denoise sweep is
currently re-traced and recompiled for every batch block, and the
memory-derived batch size varies the block shape, which multiplies distinct
compilations.  Fix: choose one fixed batch size per orientation, pad the last
block to that size, and cache one jitted vmapped function per (shape,
device).  Add an out-of-memory halve-and-retry and an absolute cap to
_auto_batch_size.  Also note: a vmapped batch runs every lane until the
slowest lane converges, so bigger batches do more total work; measure the
per-lane iteration spread before letting the batch grow at full scale.

**API.**  recon() takes devices=None (the vocabulary of mbirjax's
configure_devices: None = all visible GPUs, or an explicit list/count).  The
parallel flag is dropped: one device IS the serial path, and the same task
order runs inline with no threads.  On the cluster the Slurm request controls
what is visible, so the default needs no parameter.  Keep --serial in
recon_4d.py as an alias for devices=1 for one release.  Update
lilly_interface.md and README, which document the old flags.

**Also in this change.**
- The per-frame MBIR initialization uses the same workers and the same
  frame-to-GPU map, so a fresh initialization runs on all GPUs and its
  compiled programs and resident data carry over to the MACE loop.
- The temporal dejitter of the forward output moves after the gather step,
  because it filters along the time axis and needs all frames.  The prox
  feedback path must keep the dejittered stack in X[0], exactly as today.
- Workers live for the whole recon() call (caches survive across
  iterations); the barrier is an explicit wait on all task futures so a
  failed task raises immediately with its task identity.
- Instrumentation: add logs/task_log.csv with one row per task (iteration,
  kind, index, device, start, end).  timing_log.csv keeps iteration_total_sec
  and consensus_change_pct; the per-agent columns are replaced by
  prox_total_sec, denoise_total_sec, and makespan_sec.

**Full-scale cautions (host side, independent of the queue).**  The dejitter
(four whole-array DCT passes per iteration) and the consensus update are CPU
work that grows ~238x from smoke to full scale and may become the bottleneck
the queue cannot fix.  First steps: pass workers=-1 to scipy.fft dct/idct,
and audit host RAM before a full-scale run (W, X, and temporaries exceed
300 GB at full scale).

**Verification.**  Exact reproduction is not possible: mbirjax draws its VCD
pixel partitions from numpy's global random generator, so runs already
differ, and thread interleaving adds more variation.  Acceptance is
therefore: (a) the fast CPU test suite, including an end-to-end serial recon
(a tiny multi-device CPU test was tried and removed: at toy problem sizes the
qGGMRF line search can hit 0/0 for some random partitions, so the test failed
for reasons unrelated to the queue); (b) at smoke scale, agreement with the
previous code within its run-to-run spread, plus a visual check.

**Measured outcome (smoke scale, 4 H100s, job 15421967, 2026-08-21).**
Iteration time fell from 22 s (fixed agents, batched denoisers) to 14.4 s
steady state; iteration 1 was 36 s (compilations).  Consensus-change values
matched the previous run within run-to-run spread, and the GIF passed visual
check.  The load balance is as designed: makespan is within about 1 s of
prox_total / 4.  The gap to the 8-9 s estimate has a measured cause: each
prox task takes 1.9-2.2 s when four run concurrently, versus about 0.85 s
alone in the previous run — contention, most likely host-side (GIL during
array staging, or PCIe), not scheduling.  Candidate next steps if it matters:
profile the host staging path (pre-pinned transfer buffers, overlapped
host-to-device copies).  Re-evaluate at full resolution first: the GPU
compute per task is much larger there, so the fixed host overhead may become
irrelevant.  Day total at smoke scale: 132 s per iteration at the start,
14.4 s now.

**Smoke-run comparison (25 frames, 8x4 downsampling, 4 MACE iterations,
4 H100s, 2026-08-21).**

| | smoke1 | smoke2 | smoke3 |
|---|---|---|---|
| Commit / code | `070f849` fixed agents, serial denoisers | `43cc6f2` batched denoisers | `4b8e85c` GPU task queue |
| Denoiser sigma | per-hyperplane (1,248 estimates) | global, 0.00747 | global, 0.00747 |
| Initialization | computed (~13 min) | cached | cached |
| **Iteration time, steady state** | **~132 s** | **~22 s** | **~14.4 s** |
| Iteration 1 (with compiles) | 131 s | 38 s | 36 s |
| Forward / prox per iteration | 39 s (serial, 1 GPU) | 21-22 s (serial, 1 GPU) | 13.4 s makespan (25 tasks on 4 GPUs) |
| Denoise per iteration | 132 / 34 / 34 s (XY/YZ/XZ) | 2.5 / 1.3 / 1.3 s | 2.9 s total |
| Consensus change, iterations 2-4 | not logged yet | 12.6 / 11.6 / 11.4 % | 12.7 / 14.0 / 11.9 % |
| Total wall time | 15.6 min | 1.9 min | 1.4 min |
| Exit code | 0 | 0 | 0 |

Notes: smoke1's wall time includes computing the initialization the other two
reused, so the iteration-time row is the honest speed comparison.  smoke1's
reconstruction is not numerically comparable to the other two (per-hyperplane
vs global sigma is an intentional algorithm change); smoke2 vs smoke3 agree
within run-to-run spread.

