"""Bounded archive lookup for three remaining held maturity events."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_v84_inventory_terminal_evidence import main

if __name__ == "__main__":
    main(cases=[("113008.SH", "电气转债", "2021-01-20~2021-02-08"),
                ("110031.SH", "航信转债", "2021-06-01~2021-06-22"),
                ("110034.SH", "九州转债", "2022-01-01~2022-01-21")],
         raw=ROOT / "data_raw/inventory_terminal_v86")
