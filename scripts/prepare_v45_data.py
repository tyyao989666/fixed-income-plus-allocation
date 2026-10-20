"""Prepare strictly local frozen v45 forecast tensors and field-level audits."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v45_data import export_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reports", type=Path, default=None)
    args = parser.parse_args()
    meta = export_dataset(args.root, args.output, args.reports)
    print(json.dumps({k: meta[k] for k in ("version", "rows", "start", "end", "array_schema", "annual_splits")}, indent=2))


if __name__ == "__main__":
    main()
