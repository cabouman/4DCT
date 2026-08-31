# Cluster validation prompt — merged MACE4DModel

Copy the block below into Claude Code on the cluster.  It is written to be self-contained for
a session with no prior context.

---

```
I need to validate a refactor on real data and real GPUs.

BACKGROUND
The 4D MACE reconstruction code used to live in this repo as mace4d.py.  It has been merged
into mbirjax as mj.MACE4DModel, and recon_4d.py is now just a driver around it.  The
refactor is fully tested on CPU but three things have never been exercised:
  1. multi-GPU concurrency (only 2 virtual CPU devices were tested locally),
  2. the real NSI preprocessing path,
  3. whether the results still match the pre-merge code at full resolution.
Your job is to establish those three, in that order, stopping if a stage fails.

PATHS
  repo        : the 4DCT checkout you are in; branch refactor_for_mbirjax, commit d25a857
                (git pull first, and confirm the commit)
  dataset     : /home/li5273/Desktop/data/Phantom_30s_Run1_Dec2024
  output root : /home/li5273/Desktop/data/output/2026/0903/mace4d/
  environment : the "mbirjax" virtual environment, which already has the updated mbirjax
                installed manually

Put each stage in its own subdirectory of the output root (smoke_serial/, smoke_multigpu/,
full/) so a failed stage never contaminates the next.  Never write into the dataset directory.

WHAT CHANGED, so you know what to look at
  - MACE4DModel is constructed from ct_model + frame parameters; the sinogram now goes to
    recon(), which returns (recon, recon_dict).
  - --max_mace_itr is now --max_iterations.
  - --stop_threshold_change_pct is NEW and defaults to 0.2, so a run can now stop before
    max_iterations.  The pre-merge code always ran all of them.
  - --weight_type is NEW, defaulting to transmission_root, which is what the model used to
    hardcode.  Leaving it alone reproduces the old behavior.
  - The init cache file is renamed init_image.npy -> init_recon.npy.  An existing cache under
    the old name is NOT found and will be recomputed (15-20 min).  If a cache from a previous
    full-resolution run exists with the same shape, copy it to the new name to save that time
    -- but only for the full run, never for the smoke stages, whose shapes differ.

STAGE 0 - environment (seconds)
  Activate the mbirjax environment.  Report:
    python -c "import mbirjax as mj; print(mj.__file__); print(hasattr(mj,'MACE4DModel'), hasattr(mj,'construct_time_frames'), hasattr(mj,'save_4d_volume_as_gif'))"
    python -c "import jax; print(jax.devices())"
    nvidia-smi
  STOP and tell me if MACE4DModel is missing -- that means the manual install did not take,
  and everything after this is meaningless.
  Also check free space under the output root: the full run writes about 19 GB.

STAGE 1 - the package's own tests (about 1-2 min, CPU)
  From the mbirjax source checkout, run:
    python -m pytest tests/test_mace4d.py tests/test_utilities.py tests/test_logging.py -q
  Expect 58 passed.  If the mbirjax source is not on this machine, skip this stage and say so.
  Note: tests/test_qggmrf.py and tests/test_pallas_kernels.py have 4 failures that predate
  this work -- ignore those two files.

STAGE 2 - single-GPU smoke on real data (minutes)
  python recon_4d.py \
    --data_path    /home/li5273/Desktop/data/Phantom_30s_Run1_Dec2024 \
    --output_path  /home/li5273/Desktop/data/output/2026/0903/mace4d/smoke_serial \
    --num_frames 3 --max_iterations 1 --stop_threshold_change_pct 0 \
    --downsample_row 8 --downsample_column 4 --serial
  Pass criteria: it completes; recon_4d_*.npy exists with shape (3, nx, ny, nz) and is all
  finite; logs/run_info.txt, timing_log.csv, task_log.csv and recon_4d.gif all exist.
  This proves the real NSI path works.  Report the recon shape and the per-iteration timing.

STAGE 3 - multi-GPU, the actual point of this exercise (10-20 min)
  Same command but WITHOUT --serial, with --num_frames 12 --max_iterations 2, on a node with
  at least 2 GPUs (4 preferred), into smoke_multigpu/.
  Pass criteria, all of which you must check explicitly:
    - it completes without hanging.  If it hangs with no output for more than ~10 minutes,
      that is the failure mode we are hunting: capture py-spy dump or the stack trace and
      STOP.  Do not just retry it.
    - the distinct device column values in logs/task_log.csv equal the number of visible GPUs.
      If every task ran on device 0, the concurrency never happened and the test is void.
    - the output is finite and its shape matches the frame count.
  Report the makespan and total time per iteration from timing_log.csv, and how many GPUs
  were visible.

STAGE 4 - full resolution (about 1-1.5 h on 4x H100)
  Only if stages 2 and 3 pass.
  python recon_4d.py \
    --data_path    /home/li5273/Desktop/data/Phantom_30s_Run1_Dec2024 \
    --output_path  /home/li5273/Desktop/data/output/2026/0903/mace4d/full \
    --max_iterations 10 --stop_threshold_change_pct 0
  I am passing --stop_threshold_change_pct 0 deliberately: it forces all 10 iterations, which
  is what the pre-merge reference run did, so the comparison below is apples to apples.
  Compare against the pre-merge reference (4x H100, job 15506035, 2026-08-25/26):
    recon shape                    (97, 260, 260, 728) float32, about 19 GB
    total wall time, cached init   1:00:34
    steady-state per iteration     320-330 s
    steady-state GPU makespan      150-160 s
  Report the same four numbers.  A large discrepancy in per-iteration time matters even if the
  images look right.

STAGE 5 - numerical agreement, the strongest evidence (optional but valuable)
  If a pre-merge full-resolution recon .npy still exists on disk, find it and compare:
    NRMSE = ||new - old|| / ||old||, plus max absolute difference,
    and the same per time frame so a single bad frame is not hidden by an average.
  Do NOT expect exact equality: mbirjax draws its pixel partitions from numpy's global RNG,
  so two runs of the same code differ slightly.  A few tenths of a percent NRMSE is expected;
  a percent or more, or one frame far worse than the others, is a real finding.
  If no pre-merge recon exists, say so rather than inventing a baseline.

HOW TO RUN
  Check first whether you are on a login node or inside an allocation (echo $SLURM_JOB_ID).
  On a login node, submit each stage with sbatch (--gres=gpu:h100:4 for stages 3 and 4) and
  poll the job; do not run a GPU job on the login node.  Inside an allocation, run directly.
  Long runs: use nohup or the scheduler, not a foreground command that dies with the session.

REPORTING
  After each stage, tell me: pass or fail, the numbers asked for, and anything surprising.
  If a stage fails, stop and show me the actual error and the last 50 lines of output -- do
  not work around it, change parameters, or move on to the next stage.
```
