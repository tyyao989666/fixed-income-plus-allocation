from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v1_full_pipeline import PROFILE_OUTPUTS, profile_cfg, run_profile  # noqa: E402
from src.v1_strategy import RISK_SLEEVES, calc_metrics  # noqa: E402


PROFILES = ["low_vol", "balanced", "high_vol"]
VARIANTS = ["risk_budget_only", "views_only", "views_trend", "full_itransformer", "equal_risk_parity"]
WINDOW_STARTS = ["2022-02-28", "2022-07-31", "2023-01-31", "2023-07-31", "2024-01-31"]


def output_template(variant: str) -> dict[str, str]:
    out = {}
    for key, value in PROFILE_OUTPUTS.items():
        path = Path(value)
        out[key] = str(path.with_name(f"{path.stem}_{variant}{path.suffix}")).replace("\\", "/")
    return out


def disable_all_overlays(cfg: dict[str, Any]) -> None:
    cfg["view_tilts"] = {k: 0.0 for k in cfg.get("view_tilts", {})}
    for overlay in cfg.setdefault("signal_overlays", {}).values():
        overlay["enabled"] = False


def set_variant(cfg: dict[str, Any], profile: str, variant: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["outputs"] = {k: v.format(profile=profile) for k, v in output_template(variant).items()}

    if variant == "full_itransformer":
        return cfg

    if variant == "risk_budget_only":
        disable_all_overlays(cfg)
        return cfg

    if variant == "views_only":
        cfg.setdefault("signal_overlays", {}).setdefault("trend_following", {})["enabled"] = False
        cfg.setdefault("signal_overlays", {}).setdefault("itransformer_rank", {})["enabled"] = False
        return cfg

    if variant == "views_trend":
        cfg.setdefault("signal_overlays", {}).setdefault("itransformer_rank", {})["enabled"] = False
        return cfg

    if variant == "equal_risk_parity":
        disable_all_overlays(cfg)
        equal = 1.0 / len(RISK_SLEEVES)
        cfg["product"]["profiles"][profile]["target_risk_contribution"] = {s: equal for s in RISK_SLEEVES}
        return cfg

    raise ValueError(f"Unknown variant: {variant}")


def run_variant(profile: str, variant: str) -> dict[str, Any]:
    base_cfg = profile_cfg(profile)
    cfg = set_variant(base_cfg, profile, variant)

    import scripts.run_v1_full_pipeline as runner

    old_profile_cfg = runner.profile_cfg
    try:
        runner.profile_cfg = lambda _: copy.deepcopy(cfg)
        result = run_profile(profile)
    finally:
        runner.profile_cfg = old_profile_cfg
    return {"profile": profile, "variant": variant, "result": result, "cfg": cfg}


def metrics_for_window(nav: pd.DataFrame, start: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).ge(pd.Timestamp(start))].copy()
    if sub.empty:
        return {"months": 0}
    sub = sub.reset_index(drop=True)
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    m = calc_metrics(sub)
    return dict(zip(m["metric"], m["value"]))


def summarize_variant(profile: str, variant: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    nav = pd.read_csv(ROOT / cfg["outputs"]["nav"])
    rows = []
    for start in WINDOW_STARTS:
        metrics = metrics_for_window(nav, start)
        row = {
            "profile": profile,
            "variant": variant,
            "window_start": start,
            "months": metrics.get("months", 0),
            "annual_return": metrics.get("annual_return", np.nan),
            "annual_vol": metrics.get("annual_vol", np.nan),
            "sharpe": metrics.get("sharpe", np.nan),
            "max_drawdown": metrics.get("max_drawdown", np.nan),
            "calmar": metrics.get("calmar", np.nan),
            "avg_turnover": metrics.get("avg_turnover", np.nan),
        }
        rows.append(row)
    return rows


def write_notes(detail: pd.DataFrame, score: pd.DataFrame) -> None:
    path = ROOT / "output" / "reports" / "v1_walkforward_ablation_notes.md"
    lines = [
        "# V1 Walk-forward And Ablation",
        "",
        "## Variant Definitions",
        "",
        "- `risk_budget_only`: 只保留产品风险预算底仓。",
        "- `views_only`: 风险预算 + MarketSkill views。",
        "- `views_trend`: 风险预算 + MarketSkill views + 3/6/12月趋势。",
        "- `full_itransformer`: 风险预算 + views + trend + iTransformer rank overlay。",
        "- `equal_risk_parity`: 等风险贡献 benchmark，关闭所有 views/overlay。",
        "",
        "## Score Summary",
        "",
        score.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Key Reading",
        "",
    ]
    full = score[score["variant"].eq("full_itransformer")]
    if not full.empty:
        wins = int(full["sharpe_win_rate_vs_equal"].iloc[0] * full["cases"].iloc[0])
        cases = int(full["cases"].iloc[0])
        lines.append(f"- `full_itransformer` 相对等风险贡献的夏普胜率为 {wins}/{cases}。")
        lines.append(f"- `full_itransformer` 平均夏普为 {full['avg_sharpe'].iloc[0]:.2f}，平均年化为 {full['avg_annual_return'].iloc[0]:.2%}。")
    lines.append("- 如果某个窗口未赢，优先看是否由权益/转债或黄金单月冲击造成，不应继续无约束加大 iTransformer tilt。")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    detail_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for profile in PROFILES:
            print(f"=== {variant} / {profile} ===", flush=True)
            result = run_variant(profile, variant)
            detail_rows.extend(summarize_variant(profile, variant, result["cfg"]))

    detail = pd.DataFrame(detail_rows)
    detail_path = ROOT / "output" / "reports" / "v1_walkforward_ablation_detail.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    equal = detail[detail["variant"].eq("equal_risk_parity")][["profile", "window_start", "sharpe", "annual_return"]]
    equal = equal.rename(columns={"sharpe": "equal_sharpe", "annual_return": "equal_annual_return"})
    joined = detail.merge(equal, on=["profile", "window_start"], how="left")
    joined["beats_equal_sharpe"] = joined["sharpe"] > joined["equal_sharpe"]
    joined["beats_equal_return"] = joined["annual_return"] > joined["equal_annual_return"]

    score = (
        joined.groupby("variant", as_index=False)
        .agg(
            cases=("sharpe", "count"),
            avg_annual_return=("annual_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            min_sharpe=("sharpe", "min"),
            worst_drawdown=("max_drawdown", "min"),
            avg_turnover=("avg_turnover", "mean"),
            sharpe_win_rate_vs_equal=("beats_equal_sharpe", "mean"),
            return_win_rate_vs_equal=("beats_equal_return", "mean"),
        )
        .sort_values(["avg_sharpe", "avg_annual_return"], ascending=False)
    )
    score_path = ROOT / "output" / "reports" / "v1_walkforward_ablation_summary.csv"
    score.to_csv(score_path, index=False, encoding="utf-8-sig")
    write_notes(detail, score)

    print(
        json.dumps(
            {
                "detail": str(detail_path.relative_to(ROOT)),
                "summary": str(score_path.relative_to(ROOT)),
                "notes": "output/reports/v1_walkforward_ablation_notes.md",
                "rows": len(detail),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
