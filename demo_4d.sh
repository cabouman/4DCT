#!/bin/bash
#
# 4D MACE CT Reconstruction — demo run script.
#
# Instructions:
#   1. Set DATA_PATH to the extracted NSI dataset directory.
#   2. Optionally adjust the flags below.
#   3. Run:  bash demo_4d.sh
#


DATA_PATH=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/
OUTPUT_PATH=./output

mkdir -p "$OUTPUT_PATH"
mkdir -p ~/4dct_logs/

python recon_4d.py \
  --data_path             "$DATA_PATH" \
  --output_path           "$OUTPUT_PATH" \
  --frames_per_rotation   6 \
  --frame_overlap_factor  2.0 \
  --max_iterations        10 \
  --weight_type           transmission_root \
  --downsample_row        1 \
  --downsample_column     1 \
  --subsample_view_factor 1 \
  2>&1 | tee ~/4dct_logs/recon_4d_run.log

# To reconstruct only the first N frames (e.g. for a quick test), add:
#   --num_frames 25 \