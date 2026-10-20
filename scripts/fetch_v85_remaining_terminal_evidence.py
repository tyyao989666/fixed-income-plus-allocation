"""Four bounded archive queries for the remaining ordinary terminal events."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_v84_inventory_terminal_evidence import main

if __name__ == "__main__":
    main(cases=[("128012.SZ", "辉丰转债", "2021-04-01~2021-04-30"),
                ("123029.SZ", "英科转债", "2025-08-01~2025-08-25"),
                ("128013.SZ", "洪涛转债", "2022-07-15~2022-08-02"),
                ("128076.SZ", "金轮转债", "2025-09-20~2025-10-20")],
         raw=ROOT / "data_raw/inventory_terminal_v85",
         selected_ids={("128012.SZ", "1209624346"): "schedule",
                       ("128012.SZ", "1209837200"): "result"})
