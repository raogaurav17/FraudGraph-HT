"""
Dataset download helper for FraudGraph.
Downloads all supported datasets from public sources.

Usage:
    uv run --project backend python data/download.py --datasets ieee_cis paysim elliptic
"""

import os
import sys
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent

DATASET_INFO = {
    "ieee_cis": {
        "name": "IEEE-CIS Fraud Detection",
        "url": "https://www.kaggle.com/c/ieee-fraud-detection/data",
        "dir": DATA_DIR / "ieee_cis",
        "files": ["train_transaction.csv", "train_identity.csv", "test_transaction.csv"],
        "instructions": """
        1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
        2. Accept competition rules and download:
           - train_transaction.csv (~450 MB)
           - train_identity.csv (~30 MB)
        3. Place files in: data/ieee_cis/
        """,
    },
    "paysim": {
        "name": "PaySim Synthetic Financial Dataset",
        "url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
        "dir": DATA_DIR / "paysim",
        "files": ["PS_20174392719_1491204439457_log.csv"],
        "instructions": """
        1. Go to https://www.kaggle.com/datasets/ealaxi/paysim1
        2. Download the CSV file (~500 MB)
        3. Place in: data/paysim/
        """,
    },
    "elliptic": {
        "name": "Elliptic Bitcoin Dataset",
        "url": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        "dir": DATA_DIR / "elliptic",
        "files": [
            "elliptic_txs_features.csv",
            "elliptic_txs_edgelist.csv",
            "elliptic_txs_classes.csv",
        ],
        "instructions": """
        1. Go to https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
        2. Download all three CSV files
        3. Place in: data/elliptic/
        """,
    },
}


def check_dataset(name: str) -> dict:
    info = DATASET_INFO[name]
    existing = []
    missing = []
    for f in info["files"]:
        path = info["dir"] / f
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            existing.append(f"{f} ({size_mb:.0f} MB)")
        else:
            missing.append(f)
    return {"existing": existing, "missing": missing}


def print_status():
    print("\n" + "=" * 60)
    print("  FraudGraph — Dataset Status")
    print("=" * 60)

    for name, info in DATASET_INFO.items():
        status = check_dataset(name)
        ok = len(status["missing"]) == 0 or len(info["files"]) == 0
        icon = "✓" if ok else "✗"
        color = "\033[92m" if ok else "\033[91m"
        reset = "\033[0m"
        print(f"\n{color}{icon}{reset} {info['name']} [{name}]")
        if status["existing"]:
            for f in status["existing"]:
                print(f"    {color}found:{reset} {f}")
        if status["missing"]:
            for f in status["missing"]:
                print(f"    \033[91mmissing:{reset} {f}")

    print("\n" + "-" * 60)
    print("Manual download:   ieee_cis, paysim, elliptic (Kaggle)")
    print("-" * 60)


def print_instructions(datasets=None):
    targets = datasets or list(DATASET_INFO.keys())
    for name in targets:
        if name not in DATASET_INFO:
            continue
        info = DATASET_INFO[name]
        print(f"\n{'=' * 50}")
        print(f"  {info['name']}")
        print(f"  URL: {info['url']}")
        print(f"{'=' * 50}")
        print(info["instructions"])
        if info["dir"]:
            info["dir"].mkdir(parents=True, exist_ok=True)
            print(f"  Directory created: {info['dir']}")


def try_kaggle_download(dataset_slug: str, output_dir: Path):
    """Attempt to download via Kaggle CLI if available."""
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(output_dir), "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Downloaded via Kaggle CLI: {output_dir}")
            return True
        else:
            print(f"  Kaggle CLI failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  Kaggle CLI not found. Install: pip install kaggle")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGraph dataset manager")
    parser.add_argument("--status", action="store_true", help="Show dataset status")
    parser.add_argument("--instructions", nargs="*", help="Show download instructions")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_INFO.keys()))
    args = parser.parse_args()

    if args.status or (not args.instructions and not args.datasets):
        print_status()

    if args.instructions is not None:
        print_instructions(args.instructions if args.instructions else None)

    # Create directories
    for name, info in DATASET_INFO.items():
        info["dir"].mkdir(parents=True, exist_ok=True)
