from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_signal import build_dataset, load_config, out_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    df = build_dataset(cfg)
    meta = json.loads(out_path(cfg, "metadata").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "dataset": cfg["outputs"]["dataset"],
                "metadata": cfg["outputs"]["metadata"],
                "rows": len(df),
                "start": meta["start"],
                "end": meta["end"],
                "seq_len": meta["seq_len"],
                "feature_count": meta["feature_count"],
                "target_sleeves": meta["target_sleeves"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
