from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asset_itransformer_signal import build_asset_dataset  # noqa: E402
from src.itransformer_signal import load_config, out_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    df = build_asset_dataset(cfg)
    meta = json.loads(out_path(cfg, "asset_metadata").read_text(encoding="utf-8"))
    print(json.dumps({"dataset": cfg["outputs"]["asset_dataset"], "rows": len(df), "start": meta["start"], "end": meta["end"], "target_assets": meta["target_assets"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
