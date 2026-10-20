from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import build_asset_weights, build_sleeve_weights, load_config


if __name__ == "__main__":
    cfg = load_config()
    sleeve_weights = build_sleeve_weights(cfg)
    asset_weights = build_asset_weights(cfg)
    print(
        "v1 monthly target weights generated: "
        f"sleeve_rows={len(sleeve_weights)}, asset_rows={len(asset_weights)}, "
        f"latest={sleeve_weights['date'].max().date()}"
    )
