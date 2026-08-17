#!/bin/bash
#SBATCH --job-name=4dct_dev
#SBATCH --output=/home/li5273/Desktop/data/output/2026/0820/dev/slurm_%j.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH --time=04:00:00
#SBATCH --partition=ai
#SBATCH --account=bouman

set -e

source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate mbirjax

mkdir -p /home/li5273/Desktop/data/output/2026/0820/dev

# ── Verify / extract the dataset ─────────────────────────────────────────────
# The demo_data copy has previously been left partially extracted (unstable
# storage during copy), so verify file counts before trusting it and
# re-extract straight from the /depot source if incomplete.
DATA_TGZ=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024.tgz
DOWNLOAD_DIR=/home/li5273/PycharmProjects/lilly_exp/nsi/demo_data
EXTRACTED_DIR=$DOWNLOAD_DIR/Phantom_30s_Run1_Dec2024
MARKER=$EXTRACTED_DIR/3014053-D00087-009_Test1_30s_400delay.nsipro
RADIOGRAPHS_DIR=$EXTRACTED_DIR/Radiographs-3014053-D00087-009_Test1_30s_400delay
EXPECTED_RADIOGRAPHS=2402

mkdir -p "$DOWNLOAD_DIR"

n_radiographs=0
if [ -f "$MARKER" ]; then
    n_radiographs=$(find "$RADIOGRAPHS_DIR" -name "*.tif" 2>/dev/null | wc -l)
fi

if [ ! -f "$MARKER" ] || [ "$n_radiographs" -ne "$EXPECTED_RADIOGRAPHS" ]; then
    echo "[INFO] Dataset missing or incomplete ($n_radiographs/$EXPECTED_RADIOGRAPHS radiographs) - extracting from $DATA_TGZ"
    STAGE=$(mktemp -d "$DOWNLOAD_DIR/.extract_XXXXXX")
    tar xzf "$DATA_TGZ" -C "$STAGE"
    rm -rf "$EXTRACTED_DIR"
    mv "$STAGE/Phantom_30s_Run1_Dec2024" "$EXTRACTED_DIR"
    rmdir "$STAGE"
fi

n_radiographs=$(find "$RADIOGRAPHS_DIR" -name "*.tif" 2>/dev/null | wc -l)
if [ ! -f "$MARKER" ] || [ "$n_radiographs" -ne "$EXPECTED_RADIOGRAPHS" ]; then
    echo "[ERROR] Dataset extraction incomplete: found $n_radiographs/$EXPECTED_RADIOGRAPHS radiographs."
    exit 1
fi
echo "[INFO] Dataset verified: $EXTRACTED_DIR ($n_radiographs radiographs)"

cd /home/li5273/PycharmProjects/4DCT
python dev_recon.py
