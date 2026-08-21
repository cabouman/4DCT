"""
Download and extract a 4DCT dataset using mbirjax.

Usage:
    python download_data.py --data_url <url_or_path> --save_path <dir>

Example:
    python download_data.py \
        --data_url /depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024.tgz \
        --save_path ./data
"""

import argparse
import os

import mbirjax as mj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract a 4DCT dataset.")
    parser.add_argument("--data_url", type=str, required=True,
                        help="URL or path to the .tgz dataset file.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory where the dataset will be extracted.")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    dataset_dir = mj.download_and_extract(args.data_url, args.save_path)
    print(f"Dataset extracted to: {os.path.abspath(dataset_dir)}")
