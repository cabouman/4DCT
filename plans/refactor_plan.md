# 4DCT Refactor Plan

## Goal

Restructure the 4DCT repo to mirror the `mbirjax_applications/nsi` layout: a production-ready Lilly script driven by a shell file, a developer exploration script with full parameter access, and a clean class-based 4D model sitting underneath both.

---

## New File Structure

```
4DCT/
├── plans/
│   ├── refactor_plan.md          ← this file
│   └── lilly_interface.md        ← parameter reference for Lilly operators
├── utils.py                      ← shared utilities (dejitter, time-frame construction, denoiser helpers)
├── model_4d.py                   ← MACE4DModel class (serial + parallel)
├── Lilly_recon.py                ← production CLI script (few argparse params)
├── dev_recon.py                  ← dev/exploration script (full parameter control)
├── test_script_4D.sh             ← shell script: edit DATA_PATH and run
├── output/                       ← gitignored
└── data/                         ← gitignored
```

---
