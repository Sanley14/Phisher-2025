"""
CLI helper to create a synthetic dataset and write it to a user-specified path.
Usage:
    python tools/create_external_dataset.py F:\\phisher_data.csv

If no path is provided, will attempt to write to F:\\phisher_data.csv
"""
import sys
from pathlib import Path

from src.data.generate_synthetic_data import create_synthetic_dataset


def main():
    out = None
    if len(sys.argv) > 1:
        out = sys.argv[1]
    else:
        out = r"F:\\phisher_data.csv"

    print(f"Creating synthetic dataset at: {out}")
    try:
        res = create_synthetic_dataset(out)
        print(f"Done. Dataset written to: {res}")
    except Exception as e:
        print(f"Failed to create dataset: {e}")


if __name__ == "__main__":
    main()
