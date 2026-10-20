from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asset_itransformer_signal import train_asset_model  # noqa: E402
from src.itransformer_signal import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    result = train_asset_model(cfg)
    result.update({
        "model": cfg["outputs"]["asset_model"],
        "predictions": cfg["outputs"]["asset_predictions"],
        "validation_report": cfg["outputs"]["asset_validation_report"],
    })
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
