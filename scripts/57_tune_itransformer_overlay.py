from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v1_full_pipeline import PROFILE_OUTPUTS, profile_cfg, run_profile  # noqa: E402


PROFILES = ["low_vol", "balanced", "high_vol"]
TILTS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
MIN_TOP_PROBS = [0.00, 0.15, 0.20, 0.25, 0.30, 0.35]


def output_template(variant: str) -> dict[str, str]:
    out = {}
    for key, value in PROFILE_OUTPUTS.items():
        path = Path(value)
        out[key] = str(path.with_name(f"{path.stem}_{variant}{path.suffix}")).replace("\\", "/")
    return out


def metrics_as_dict(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def candidate_cfg(profile: str, tilt: float, min_top_prob: float, variant: str) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    cfg["outputs"] = {k: v.format(profile=profile) for k, v in output_template(variant).items()}
    overlay = cfg.setdefault("signal_overlays", {}).setdefault("itransformer_rank", {})
    overlay["enabled"] = True
    overlay["tilt"] = float(tilt)
    overlay["min_top_prob"] = float(min_top_prob)
    return cfg


def run_candidate(profile: str, tilt: float, min_top_prob: float, variant: str) -> dict[str, Any]:
    cfg = candidate_cfg(profile, tilt, min_top_prob, variant)

    import scripts.run_v1_full_pipeline as runner

    old_profile_cfg = runner.profile_cfg
    try:
        runner.profile_cfg = lambda _: copy.deepcopy(cfg)
        result = run_profile(profile)
    finally:
        runner.profile_cfg = old_profile_cfg

    metrics = metrics_as_dict(ROOT / cfg["outputs"]["metrics"])
    return {
        "variant": variant,
        "profile": profile,
        "tilt": tilt,
        "min_top_prob": min_top_prob,
        "annual_return": metrics.get("annual_return"),
        "annual_vol": metrics.get("annual_vol"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "calmar": metrics.get("calmar"),
        "avg_turnover": metrics.get("avg_turnover"),
        "backtest_months": result.get("backtest_months"),
    }


def aggregate_scores(rows: pd.DataFrame) -> pd.DataFrame:
    agg = (
        rows.groupby(["tilt", "min_top_prob"], as_index=False)
        .agg(
            avg_annual_return=("annual_return", "mean"),
            min_annual_return=("annual_return", "min"),
            avg_sharpe=("sharpe", "mean"),
            min_sharpe=("sharpe", "min"),
            worst_drawdown=("max_drawdown", "min"),
            avg_calmar=("calmar", "mean"),
            avg_turnover=("avg_turnover", "mean"),
        )
    )
    # Conservative objective: reward return and Sharpe, penalize turnover and drawdown.
    agg["score"] = (
        agg["avg_sharpe"]
        + 2.0 * agg["avg_annual_return"]
        + 0.15 * agg["avg_calmar"]
        - 0.50 * agg["avg_turnover"]
        + 2.0 * agg["worst_drawdown"]
    )
    return agg.sort_values(["score", "avg_sharpe", "avg_annual_return"], ascending=False)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for tilt in TILTS:
        for min_top_prob in MIN_TOP_PROBS:
            variant = f"itf_t{int(tilt * 1000):03d}_p{int(min_top_prob * 100):02d}"
            print(f"\n=== candidate {variant} ===", flush=True)
            for profile in PROFILES:
                rows.append(run_candidate(profile, tilt, min_top_prob, variant))

    detail = pd.DataFrame(rows)
    detail_path = ROOT / "output" / "reports" / "itransformer_overlay_tuning_detail.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = aggregate_scores(detail)
    summary_path = ROOT / "output" / "reports" / "itransformer_overlay_tuning_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best = summary.iloc[0].to_dict()

    print(
        json.dumps(
            {
                "detail": str(detail_path.relative_to(ROOT)),
                "summary": str(summary_path.relative_to(ROOT)),
                "best": best,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
