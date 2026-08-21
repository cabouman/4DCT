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
├── mace4d.py                     ← MACE4DModel class + all 4D helpers
├── recon_4d.py                   ← command-line reconstruction driver (all params as flags)
├── demo_4d.sh                    ← shell script: edit DATA_PATH and run
├── output/                       ← gitignored
└── data/                         ← gitignored
```

---
