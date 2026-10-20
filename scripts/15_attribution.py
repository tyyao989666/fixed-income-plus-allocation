from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import load_config, run_attribution


if __name__ == "__main__":
    cfg = load_config()
    attribution = run_attribution(cfg)
    print(f"v1 attribution generated: rows={len(attribution)}")
