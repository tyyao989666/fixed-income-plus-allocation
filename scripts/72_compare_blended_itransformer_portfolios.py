from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v1_full_pipeline import PROFILE_OUTPUTS, profile_cfg  # noqa: E402
from src.v1_strategy import run_attribution, run_backtest, run_robustness  # noqa: E402


def metric_map(path: Path) -> dict[str, float]:
    return pd.read_csv(path).set_index("metric")["value"].to_dict()


def main() -> None:
    rows = []
    for profile in ["low_vol", "balanced", "high_vol"]:
        cfg = copy.deepcopy(profile_cfg(profile))
        cfg["signal_overlays"]["itransformer_rank"]["predictions_path"] = "data_processed/itransformer_predictions_daily_blended_monthly.parquet"
        cfg["outputs"] = {key: value.replace("v1_{profile}", "v1_{profile}_blended_itransformer").format(profile=profile) for key, value in PROFILE_OUTPUTS.items()}
        run_backtest(cfg)
        run_attribution(cfg)
        run_robustness(cfg)
        candidate = metric_map(ROOT / cfg["outputs"]["metrics"])
        baseline = metric_map(ROOT / f"output/reports/v1_{profile}_backtest_metrics.csv")
        rows.append({
            "profile": profile,
            "baseline_annual_return": baseline["annual_return"],
            "candidate_annual_return": candidate["annual_return"],
            "delta_annual_return": candidate["annual_return"] - baseline["annual_return"],
            "baseline_sharpe": baseline["sharpe"],
            "candidate_sharpe": candidate["sharpe"],
            "delta_sharpe": candidate["sharpe"] - baseline["sharpe"],
            "baseline_max_drawdown": baseline["max_drawdown"],
            "candidate_max_drawdown": candidate["max_drawdown"],
            "delta_max_drawdown": candidate["max_drawdown"] - baseline["max_drawdown"],
            "baseline_turnover": baseline["avg_turnover"],
            "candidate_turnover": candidate["avg_turnover"],
        })
    out = pd.DataFrame(rows)
    path = ROOT / "output" / "reports" / "v1_blended_itransformer_portfolio_comparison.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(json.dumps(out.to_dict(orient="records"), ensure_ascii=False))


if __name__ == "__main__":
    main()
