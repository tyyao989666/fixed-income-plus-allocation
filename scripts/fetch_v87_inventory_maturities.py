"""Bounded queries for the five ordinary-maturity investigation candidates."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_v84_inventory_terminal_evidence import main

if __name__ == "__main__":
    main(cases=[("127003.SZ", "海印转债", "2022-05-25~2022-06-15"),
                ("127016.SZ", "鲁泰转债", "2026-03-25~2026-04-16"),
                ("110070.SH", "凌钢转债", "2026-04-01~2026-04-20"),
                ("123049.SZ", "维尔转债", "2026-04-01~2026-04-20"),
                ("113589.SH", "天创转债", "2026-06-10~2026-06-30")],
         raw=ROOT / "data_raw/inventory_terminal_v87")
