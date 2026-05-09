#!/usr/bin/env python3
"""
LongMemEval Full Dataset Download Script

Downloads the xiaowu0162/longmemeval-cleaned dataset from HuggingFace
into the data/ directory.

Usage:
    python download_data.py                # Use huggingface-cli (recommended)
    python download_data.py --method git   # Use git lfs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
DATA_DIR = Path(__file__).parent / "data"


def download_hf_cli() -> None:
    """Download the dataset using the HuggingFace CLI.

    Exits with code 1 if the download fails or the CLI is not installed.
    """
    print(f"Downloading {DATASET_REPO} to {DATA_DIR}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli", "download", DATASET_REPO,
        "--repo-type", "dataset",
        "--local-dir", str(DATA_DIR),
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Download complete. Data saved to: {DATA_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure huggingface_hub is installed: pip install huggingface_hub")
        print("  2. Log in to HuggingFace: huggingface-cli login")
        print("  3. Check network connectivity")
        sys.exit(1)
    except FileNotFoundError:
        print("huggingface-cli not found. Install with: pip install huggingface_hub")
        sys.exit(1)


def download_git_lfs() -> None:
    """Download the dataset by cloning the HuggingFace repo with Git LFS.

    Exits with code 1 if Git LFS is unavailable or the clone fails.
    """
    print(f"Cloning {DATASET_REPO} to {DATA_DIR}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://huggingface.co/datasets/{DATASET_REPO}"

    lfs_cmd = ["git", "lfs", "install"]
    try:
        subprocess.run(lfs_cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("git-lfs not available. Install with: apt install git-lfs")
        print("Then run: git lfs install")
        sys.exit(1)

    clone_cmd = ["git", "clone", repo_url, str(DATA_DIR)]
    try:
        subprocess.run(clone_cmd, check=True)
        print(f"Download complete. Data saved to: {DATA_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        sys.exit(1)


def main() -> None:
    """Entry point: parse arguments and dispatch to the chosen download method."""
    parser = argparse.ArgumentParser(description="Download LongMemEval dataset from HuggingFace")
    parser.add_argument(
        "--method", choices=["hf-cli", "git"], default="hf-cli",
        help="Download method: hf-cli (huggingface-cli) or git (git lfs)",
    )
    args = parser.parse_args()

    if args.method == "hf-cli":
        download_hf_cli()
    else:
        download_git_lfs()


if __name__ == "__main__":
    main()
