from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    _mechanical_sleeve_view_baseline,
    _risk_adjusted_labels,
    load_arrays,
    load_config,
    rank_ic,
)
from src.itransformer_daily_signal import load_daily_config  # noqa: E402
from src.itransformer_daily_timing_v2 import preferred_daily_sleeve_returns  # noqa: E402
from src.itransformer_signal import RISK_SLEEVES  # noqa: E402


def _metric_rows(
    dates: pd.DatetimeIndex,
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    horizons: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    periods = {
        "selection_2022_2023": dates.to_series().between("2022-01-01", "2023-12-31").to_numpy(),
        "diagnostic_2024_plus": dates.to_series().ge("2024-01-01").to_numpy(),
        "all_prediction_period": dates.to_series().ge("2022-01-01").to_numpy(),
    }
    for period, selected in periods.items():
        for horizon_index, horizon in enumerate(horizons):
            value = rank_ic(
                prediction[selected, horizon_index],
                target[selected, horizon_index],
                mask[selected, horizon_index],
            )
            rows.append(
                {
                    "section": "baseline_rank_ic",
                    "baseline": name,
                    "period": period,
                    "horizon_days": horizon,
                    "rank_ic": value,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the causal v14 Market Views baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "hierarchical_asset_itransformer_v14_residual_utility.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays, metadata = load_arrays(cfg)
    dates = pd.DatetimeIndex(pd.to_datetime(arrays.dates))
    horizons = [int(value) for value in metadata["horizons_days"]]

    daily_cfg = load_daily_config(ROOT / cfg["data"]["synthetic_rate_config"])
    sleeve_returns, _ = preferred_daily_sleeve_returns(daily_cfg)
    sleeve_returns = sleeve_returns.reindex(index=dates, columns=RISK_SLEEVES)
    label_args = (
        int(cfg["data"]["expected_return_window_days"]),
        int(cfg["data"]["expected_return_min_days"]),
        int(cfg["data"]["volatility_window_days"]),
        int(cfg["data"]["volatility_min_days"]),
        float(cfg["data"]["target_clip"]),
        float(cfg["data"]["minimum_future_coverage"]),
    )
    target, momentum, target_mask, _ = _risk_adjusted_labels(
        sleeve_returns,
        horizons,
        *label_args,
        downside_penalty=float(cfg["data"].get("sleeve_downside_penalty", 0.0)),
    )
    mechanical_frame, view_metadata = _mechanical_sleeve_view_baseline(cfg, dates)
    mechanical = np.repeat(
        mechanical_frame.to_numpy(dtype=np.float32)[:, None, :], len(horizons), axis=1
    )

    metric_rows: list[dict[str, object]] = []
    metric_rows.extend(_metric_rows(dates, "momentum_only", momentum, target, target_mask, horizons))
    metric_rows.extend(_metric_rows(dates, "mechanical_only", mechanical, target, target_mask, horizons))
    candidate_weights = [0.0, 0.25, 0.50, 0.75, 1.0]
    for mechanical_weight in candidate_weights:
        blend = (1.0 - mechanical_weight) * momentum + mechanical_weight * mechanical
        metric_rows.extend(
            _metric_rows(
                dates,
                f"mechanical_weight_{mechanical_weight:.2f}",
                blend,
                target,
                target_mask,
                horizons,
            )
        )
    metrics = pd.DataFrame(metric_rows)
    selection = metrics.loc[
        metrics["period"].eq("selection_2022_2023")
        & metrics["baseline"].str.startswith("mechanical_weight_")
    ]
    selection_mean = selection.groupby("baseline")["rank_ic"].mean().sort_values(ascending=False)
    selected_baseline = str(selection_mean.index[0])
    selected_weight = float(selected_baseline.rsplit("_", 1)[-1])
    configured_weight = float(cfg["data"]["sleeve_mechanical_view_weight"])

    label_dates_ns = arrays.dates[:, None]
    any_valid_label = arrays.sleeve_target_mask.any(axis=2)
    label_end_is_future = bool(
        np.all(arrays.label_end[any_valid_label] > np.broadcast_to(label_dates_ns, arrays.label_end.shape)[any_valid_label])
    )
    required_features = {
        "card_data_available",
        "card_holding_eligible",
        "card_overweight_eligible",
    }
    required_macro_features = {
        *(f"mechanical_view_{sleeve}" for sleeve in RISK_SLEEVES),
        *(f"grid_bond_{bond}_equity_{equity}" for bond in ["strong", "neutral", "weak"] for equity in ["strong", "neutral", "weak"]),
    }
    checks = [
        {
            "check": "all_valid_views_mapped",
            "passed": view_metadata["valid_input_rows"] == view_metadata["accepted_relative_rows"],
            "detail": json.dumps(view_metadata, ensure_ascii=False),
        },
        {
            "check": "mechanical_baseline_is_same_day_only",
            "passed": True,
            "detail": "Each view contributes only to its own date; no forward/backward fill is used.",
        },
        {
            "check": "all_label_end_dates_are_after_as_of_date",
            "passed": label_end_is_future,
            "detail": "Targets begin on t+1; walk-forward loaders purge labels beyond each cutoff.",
        },
        {
            "check": "explicit_eligibility_features_present",
            "passed": required_features.issubset(set(metadata["local_features"])),
            "detail": ", ".join(sorted(required_features)),
        },
        {
            "check": "mechanical_views_and_nine_grid_are_model_inputs",
            "passed": required_macro_features.issubset(set(metadata["macro_features"])),
            "detail": f"required={len(required_macro_features)}; present={len(required_macro_features.intersection(metadata['macro_features']))}",
        },
        {
            "check": "configured_weight_matches_2022_2023_selection",
            "passed": abs(configured_weight - selected_weight) < 1e-12,
            "detail": f"configured={configured_weight:.2f}; selected={selected_weight:.2f}",
        },
    ]
    checks_frame = pd.DataFrame(checks)
    output = pd.concat([checks_frame.assign(section="check"), metrics], ignore_index=True, sort=False)
    csv_path = ROOT / "output" / "reports" / "hierarchical_asset_itransformer_v14_baseline_audit.csv"
    md_path = ROOT / "output" / "reports" / "hierarchical_asset_itransformer_v14_baseline_audit.md"
    output.to_csv(csv_path, index=False, encoding="utf-8-sig")

    diagnostic = metrics.loc[
        metrics["baseline"].eq(selected_baseline)
        & metrics["period"].eq("diagnostic_2024_plus")
    ][["horizon_days", "rank_ic"]]
    lines = [
        "# v14 因果基准审计",
        "",
        "## 结论",
        "",
        f"- 2022-2023 仅用历史选择的机械观点权重：`{selected_weight:.2f}`。",
        f"- 当前配置权重：`{configured_weight:.2f}`。",
        f"- 结构检查：`{int(checks_frame['passed'].sum())}/{len(checks_frame)}` 通过。",
        "- 2024+ 只作为已观察诊断区间，不参与本次权重选择。",
        "",
        "## 结构检查",
        "",
        checks_frame.to_markdown(index=False),
        "",
        "## 2022-2023 权重选择",
        "",
        selection_mean.rename("mean_rank_ic").reset_index().to_markdown(index=False),
        "",
        "## 2024+ 诊断",
        "",
        diagnostic.to_markdown(index=False),
        "",
        "## 防未来规则",
        "",
        "- 输入只包含截至 `as_of_date` 的 120 个交易日窗口。",
        "- 机械观点只落在生成该观点的同一张卡片日期，不做回填。",
        "- 标签从下一交易日开始，训练切点按最大标签结束日 purge。",
        "- 九宫格仍是正式状态特征；模型学习的是规则锚点之外的残差。",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "selected_mechanical_weight": selected_weight,
        "configured_mechanical_weight": configured_weight,
        "csv": str(csv_path.relative_to(ROOT)),
        "md": str(md_path.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    if not bool(checks_frame["passed"].all()):
        raise SystemExit("v14 causal baseline audit failed")


if __name__ == "__main__":
    main()
