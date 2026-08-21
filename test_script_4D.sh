#!/bin/bash
#
# 4D MACE CT Reconstruction — Lilly production run script.
#
# Instructions:
#   1. Set DATA_PATH to the extracted NSI dataset directory.
#   2. Optionally adjust the flags below.
#   3. Run:  bash test_script_4D.sh
#


DATA_PATH=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/
OUTPUT_PATH=./output

mkdir -p "$OUTPUT_PATH"
mkdir -p ~/4dct_logs/

python Lilly_recon.py \
  --data_path             "$DATA_PATH" \
  --output_path           "$OUTPUT_PATH" \
  --angle_span_per_recon  120.0 \
  --angle_advancing       60.0 \
  --max_mace_itr          10 \
  --num_frames            25 \
  --downsample_row        1 \
  --downsample_column     1 \
  --subsample_view_factor 1 \
  `# --resume /path/to/init_image.npy` \
  2>&1 | tee ~/4dct_logs/lilly_run.log
