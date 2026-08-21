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
