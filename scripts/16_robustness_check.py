from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import load_config, run_robustness


if __name__ == "__main__":
    cfg = load_config()
    robustness = run_robustness(cfg)
    print(f"v1 robustness summary generated: rows={len(robustness)}")
