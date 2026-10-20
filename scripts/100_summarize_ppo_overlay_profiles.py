from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    rows = []
    for profile in ["low_vol", "balanced", "high_vol"]:
        path = ROOT / "output" / "reports" / f"daily_ppo_overlay_evaluation_{profile}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for split, split_data in data.items():
            noop = split_data.get("noop", {})
            model = split_data.get("model", {})
            row = {
                "profile": profile,
                "split": split,
                "noop_mean_reward": noop.get("mean_reward"),
                "model_mean_reward": model.get("mean_reward"),
                "delta_mean_reward": None,
                "noop_mean_gross_return": noop.get("mean_gross_return"),
                "model_mean_gross_return": model.get("mean_gross_return"),
                "delta_mean_gross_return": None,
                "noop_mean_turnover": noop.get("mean_turnover"),
                "model_mean_turnover": model.get("mean_turnover"),
                "delta_mean_turnover": None,
            }
            if row["model_mean_reward"] is not None and row["noop_mean_reward"] is not None:
                row["delta_mean_reward"] = row["model_mean_reward"] - row["noop_mean_reward"]
            if row["model_mean_gross_return"] is not None and row["noop_mean_gross_return"] is not None:
                row["delta_mean_gross_return"] = row["model_mean_gross_return"] - row["noop_mean_gross_return"]
            if row["model_mean_turnover"] is not None and row["noop_mean_turnover"] is not None:
                row["delta_mean_turnover"] = row["model_mean_turnover"] - row["noop_mean_turnover"]
            rows.append(row)
    out = pd.DataFrame(rows)
    path = ROOT / "output" / "reports" / "daily_ppo_overlay_three_profile_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(json.dumps({"summary": str(path.relative_to(ROOT)), "rows": len(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
