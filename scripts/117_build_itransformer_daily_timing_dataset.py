from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import load_daily_config  # noqa: E402
from src.itransformer_daily_timing import build_timing_dataset, timing_path  # noqa: E402


def main() -> None:
    cfg = load_daily_config()
    frame = build_timing_dataset(cfg)
    metadata = json.loads(timing_path(cfg, "timing_metadata").read_text(encoding="utf-8"))
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
