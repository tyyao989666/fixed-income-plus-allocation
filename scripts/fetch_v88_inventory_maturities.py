"""Fetch the three newly exposed maturity candidates without changing older evidence."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_v84_inventory_terminal_evidence import main

if __name__ == "__main__":
    main(cases=[("113545.SH", "金能转债", "2025-09-20~2025-10-20"),
                ("127015.SZ", "希望转债", "2025-12-20~2026-01-15"),
                ("128134.SZ", "鸿路转债", "2026-09-20~2026-10-14")],
         raw=ROOT / "data_raw/inventory_terminal_v88")
