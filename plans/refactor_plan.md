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

### Next: one GPU task queue for all agent work (planned 2026-08-21)

The forward model now limits the iteration time.  Its 25 proximal maps run in
series on one GPU while the other three GPUs sit mostly idle.  The proximal
maps are independent of each other, so they can run on different GPUs at the
same time.

**Design.**  Each MACE iteration produces 28 independent tasks: one proximal
map per time frame (25 tasks) and one batched denoise per prior orientation
(3 tasks).  A pool of N worker threads executes the tasks, one worker per GPU.
Every task runs entirely on one GPU; no task is split across GPUs.  The old
fixed roles (one forward agent, three prior agents) disappear.

```
Task list (assignment computed once, reused every iteration; N = 4 shown)

  prox frame 0  -> GPU 0        denoise XY-t -> GPU 1
  prox frame 1  -> GPU 1        denoise YZ-t -> GPU 2
  prox frame 2  -> GPU 2        denoise XZ-t -> GPU 3
  prox frame 3  -> GPU 3
  prox frame 4  -> GPU 0
  ...
  prox frame 24 -> GPU 0

  GPU 0 worker: prox 0, 4, 8, 12, 16, 20, 24
  GPU 1 worker: prox 1, 5, ... 21, then denoise XY-t
  GPU 2 worker: prox 2, 6, ... 22, then denoise YZ-t
  GPU 3 worker: prox 3, 7, ... 23, then denoise XZ-t
       |
       v
  barrier: all tasks done -> gather frames -> dejitter -> consensus update
```

**Sticky assignment.**  The task-to-GPU assignment is computed once, round
robin, and reused for every iteration.  The reason is compilation cost.  Each
frame's cone-beam model is pinned to one GPU, and mbirjax compiles its
projection kernels for that device.  If frame 7 ran on GPU 2 in one iteration
and on GPU 0 in the next, the code would need a second pinned copy of the
model and a second compilation.  A fully dynamic queue could trigger up to
25 x N compilations before every (frame, GPU) pair is warm.  With a sticky
assignment each pair compiles exactly once.  The cost of stickiness is small:
the tasks are nearly uniform in size, so a fixed round robin finishes within
about one task length of a fully dynamic queue.

Upgrade path: if profiling later shows the task times are far from uniform,
first check whether mbirjax shares compiled kernels between model instances of
the same shape on one device.  If it does, remove the stickiness and the queue
becomes fully dynamic.  The surrounding structure does not change.

**Number of GPUs.**  Use all visible GPUs (jax.devices('gpu')).  On the
cluster the Slurm request already controls how many GPUs are visible, so no
separate parameter is needed.  The num_forward_gpus / num_prior_gpus idea is
dropped; the queue replaces it.  parallel=False keeps the current
single-device serial path.

**Also in this change.**
- The per-frame MBIR initialization uses the same queue, so a fresh
  initialization runs on all GPUs instead of one.
- The temporal dejitter of the forward output moves after the gather step,
  because it filters along the time axis and needs all frames.
- One thread pool is created per recon() call instead of per iteration, so
  the per-thread denoiser caches survive across iterations.
- The timing log keeps its columns.  agent_0_forward_sec becomes the wall
  time from iteration start until the last proximal-map task finishes, and
  the prior columns keep their per-orientation times.

**Verification.**  The assignment does not change any computation, so the
queued version must reproduce the current results exactly.  Check on the
smoke configuration with the cached initialization.  Expected iteration time
at smoke scale: about 27 seconds of total work spread over 4 GPUs, so about
7 seconds per iteration, versus 23 seconds now.

