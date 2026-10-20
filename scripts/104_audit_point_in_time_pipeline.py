from __future__ import annotations

import json
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import build_daily_feature_frame, load_daily_config  # noqa: E402
from src.itransformer_signal import build_feature_frame, load_config  # noqa: E402
from src.point_in_time import transform_point_in_time_features  # noqa: E402
from src.v1_strategy import (  # noqa: E402
    V1Paths,
    apply_selected_profile_overrides,
    build_sleeve_returns,
    calibrate_market_view_confidence,
    load_config as load_v1_config,
    month_end_views,
)


def add(rows: list[dict], check: str, status: str, value: object, detail: str) -> None:
    rows.append({"check": check, "status": status, "value": value, "detail": detail})


def causal_prefix_check(name: str, frame: pd.DataFrame, cols: list[str], preprocessing: dict, rows: list[dict]) -> None:
    cutoff = max(2, int(len(frame) * 0.70))
    full, full_cols = transform_point_in_time_features(frame, cols, preprocessing)
    prefix, prefix_cols = transform_point_in_time_features(frame.iloc[:cutoff].copy(), cols, preprocessing)
    same_schema = full_cols == prefix_cols
    full = full.iloc[:cutoff]
    diff = np.abs(full.to_numpy(dtype=float) - prefix.to_numpy(dtype=float))
    max_diff = float(np.nanmax(diff)) if diff.size else 0.0
    finite = bool(np.isfinite(full.to_numpy(dtype=float)).all())
    status = "pass" if max_diff <= 1e-12 and finite and same_schema else "fail"
    add(rows, f"{name}_future_append_invariance", status, max_diff, "Appending future rows must not change any historical standardized value.")
    add(rows, f"{name}_finite_inputs", "pass" if finite else "fail", finite, "Model inputs must contain no NaN or infinity after causal normalization.")
    add(rows, f"{name}_stable_feature_schema", "pass" if same_schema else "fail", same_schema, "Full and prefix transforms must produce identical feature definitions.")


def dataset_check(path: Path, rows: list[dict]) -> None:
    if not path.exists():
        add(rows, f"dataset_{path.stem}", "warn", "missing", "Rebuild the dataset before training.")
        return
    df = pd.read_parquet(path)
    x_cols = [c for c in df.columns if c.startswith("x_t")]
    finite = bool(np.isfinite(df[x_cols].to_numpy(dtype=float)).all()) if x_cols else False
    ordered = bool((pd.to_datetime(df["label_date"]) > pd.to_datetime(df["as_of_date"])).all())
    add(rows, f"dataset_{path.stem}_finite", "pass" if finite else "fail", finite, f"rows={len(df)}, features={len(x_cols)}")
    add(rows, f"dataset_{path.stem}_label_after_decision", "pass" if ordered else "fail", ordered, "Labels must be strictly later than decision dates.")


def static_source_checks(rows: list[dict]) -> None:
    files = [
        ROOT / "src" / "itransformer_signal.py",
        ROOT / "src" / "itransformer_daily_signal.py",
        ROOT / "src" / "asset_itransformer_signal.py",
    ]
    text = "\n".join(p.read_text(encoding="utf-8") for p in files)
    add(rows, "no_backward_fill", "pass" if ".bfill(" not in text else "fail", ".bfill(" not in text, "Backward fill is forbidden in model feature pipelines.")

    builder = (ROOT / "scripts" / "47_build_v43_basic_data.py").read_text(encoding="utf-8")
    release_aware = "left_on=\"date\"" in builder and "right_on=\"visible_date\"" in builder and "direction=\"backward\"" in builder
    add(rows, "macro_release_date_asof", "pass" if release_aware else "fail", release_aware, "Monthly CPI used by real-rate features must enter on its visible/effective date.")

    ohlcv_causal = "causal_rolling_quantile_clip" in text
    add(rows, "ohlcv_causal_winsorization", "pass" if ohlcv_causal else "fail", ohlcv_causal, "OHLCV clipping must use trailing, not full-sample, quantiles.")

    timing_backtest = (ROOT / "scripts" / "103_backtest_daily_timing_cards.py").read_text(encoding="utf-8")
    t_plus_one = "decisions[position - 1]" in timing_backtest
    add(rows, "daily_timing_t_plus_one", "pass" if t_plus_one else "fail", t_plus_one, "A close-of-day timing card may first affect the next trading day's return.")

    v1_strategy = (ROOT / "src" / "v1_strategy.py").read_text(encoding="utf-8")
    asset_momentum_asof = 'returns["date"].le(pd.Timestamp(date))' in v1_strategy
    add(rows, "monthly_asset_momentum_asof", "pass" if asset_momentum_asof else "fail", asset_momentum_asof, "Asset momentum may only read returns dated on or before the allocation decision.")
    sleeve_rotation_asof = "hist = sleeve_ret.loc[:date].sort_index()" in v1_strategy
    add(rows, "monthly_sleeve_rotation_asof", "pass" if sleeve_rotation_asof else "fail", sleeve_rotation_asof, "Risk-on sleeve rotation may only read sleeve returns through the decision date.")

    strategy_cfg = (ROOT / "config" / "v1_strategy_config.yaml").read_text(encoding="utf-8")
    monthly_lagged = "signal_lag_periods: 1" in strategy_cfg
    add(rows, "monthly_allocation_t_plus_one", "pass" if monthly_lagged else "fail", monthly_lagged, "Month-end target weights must first apply to the following month's return.")
    aligned_sleeve_returns = "sleeve_return_aggregation: preferred_asset_weights" in strategy_cfg
    add(rows, "formal_sleeve_return_alignment", "pass" if aligned_sleeve_returns else "fail", aligned_sleeve_returns, "Risk-budget and EP sleeve returns must match each profile's formal asset anchor.")


def view_preprocessing_checks(rows: list[dict]) -> None:
    cfg = load_v1_config()
    raw_cfg = copy.deepcopy(cfg)
    raw_cfg.setdefault("signal_overlays", {}).setdefault("view_preprocessing", {})["mode"] = "raw_month"
    raw = month_end_views(V1Paths(), raw_cfg)
    formal = month_end_views(V1Paths(), cfg)
    asof_ok = bool((pd.to_datetime(formal["date"]) <= pd.to_datetime(formal["month"])).all())
    add(rows, "monthly_view_preprocessing_asof", "pass" if asof_ok else "fail", asof_ok, "A monthly view aggregate may only contain cards observed on or before that month-end decision.")

    keys = ["month", "source_asset", "target_asset"]
    unique_pairs = not bool(formal.duplicated(keys).any())
    add(rows, "monthly_view_pair_deduplication", "pass" if unique_pairs else "fail", unique_pairs, "Formal monthly preprocessing must emit at most one confidence per directional asset pair.")

    expected = raw.groupby(keys, dropna=False)["confidence"].mean().sort_index()
    actual = formal.set_index(keys)["confidence"].sort_index()
    same_index = expected.index.equals(actual.index)
    max_diff = float((expected - actual).abs().max()) if same_index and not expected.empty else np.inf
    mean_ok = bool(same_index and max_diff <= 1e-12)
    add(rows, "monthly_view_confidence_mean", "pass" if mean_ok else "fail", max_diff, "Formal confidence must equal the mean of already-observed daily confidences for the same directional pair.")

    reliability_cfg = copy.deepcopy(cfg)
    reliability_cfg["signal_overlays"]["view_reliability"]["enabled"] = True
    sleeve_ret = build_sleeve_returns(V1Paths(), reliability_cfg)
    cutoff = pd.Timestamp(sleeve_ret.index[-12])
    full_calibrated = calibrate_market_view_confidence(formal, sleeve_ret, reliability_cfg)
    prefix_calibrated = calibrate_market_view_confidence(formal, sleeve_ret.loc[:cutoff], reliability_cfg)
    compare_cols = ["month", "source_asset", "target_asset"]
    value_cols = ["view_reliability_factor", "view_reliability_hit_rate", "confidence"]
    left = full_calibrated[full_calibrated["month"].le(cutoff)][compare_cols + value_cols]
    right = prefix_calibrated[prefix_calibrated["month"].le(cutoff)][compare_cols + value_cols]
    merged = left.merge(right, on=compare_cols, how="outer", suffixes=("_full", "_prefix"), indicator=True)
    reliability_diff = 0.0
    if not merged.empty and merged["_merge"].eq("both").all():
        reliability_diff = max(float((merged[f"{col}_full"] - merged[f"{col}_prefix"]).abs().max()) for col in value_cols)
    else:
        reliability_diff = np.inf
    reliability_ok = bool(reliability_diff <= 1e-12)
    add(rows, "monthly_view_reliability_future_append_invariance", "pass" if reliability_ok else "fail", reliability_diff, "Appending later sleeve returns must not change any reliability factor already available at the cutoff month.")

    min_observations = int(reliability_cfg["signal_overlays"]["view_reliability"]["min_observations"])
    immature = full_calibrated["view_reliability_observations"].lt(min_observations)
    immature_neutral = bool(full_calibrated.loc[immature, "view_reliability_factor"].eq(1.0).all())
    add(rows, "monthly_view_reliability_min_history_neutral", "pass" if immature_neutral else "fail", immature_neutral, "Reliability calibration must remain neutral until the configured minimum history is available.")

    channels: dict[str, str] = {}
    for profile in ["low_vol", "balanced", "high_vol"]:
        profile_cfg = copy.deepcopy(cfg)
        profile_cfg["product"]["selected_profile"] = profile
        profile_cfg = apply_selected_profile_overrides(profile_cfg)
        application = profile_cfg["signal_overlays"]["view_application"]
        direct = bool(application.get("direct_tilt_enabled", True))
        ep_mean = bool(application.get("ep_mean_enabled", True))
        channels[profile] = f"direct={direct},ep_mean={ep_mean}"
    single_channel = all(value in {"direct=True,ep_mean=False", "direct=False,ep_mean=True"} for value in channels.values())
    add(rows, "formal_market_view_single_transmission", "pass" if single_channel else "fail", json.dumps(channels, ensure_ascii=False), "Each formal product must use exactly one Market View transmission channel to prevent duplicate counting.")


def main() -> None:
    rows: list[dict] = []
    static_source_checks(rows)
    view_preprocessing_checks(rows)

    monthly_cfg = load_config()
    monthly, monthly_cols = build_feature_frame(monthly_cfg)
    causal_prefix_check("monthly", monthly, monthly_cols, monthly_cfg["data"].get("preprocessing", {}), rows)

    daily_cfg = load_daily_config()
    daily, daily_cols = build_daily_feature_frame(daily_cfg)
    causal_prefix_check("daily", daily, daily_cols, daily_cfg["data"].get("preprocessing", {}), rows)
    transformed, transformed_cols = transform_point_in_time_features(daily, daily_cols, daily_cfg["data"].get("preprocessing", {}))
    long_anchors = [c for c in transformed_cols if "__pct_756" in c or "__pct_1260" in c]
    deltas = [c for c in transformed_cols if "__delta_20" in c]
    add(rows, "daily_multiscale_method", "pass" if daily_cfg["data"].get("preprocessing", {}).get("method") == "causal_multiscale_robust" else "fail", daily_cfg["data"].get("preprocessing", {}).get("method"), "Daily timing model must use feature-specific windows rather than one universal scaler.")
    add(rows, "daily_long_state_anchors", "pass" if long_anchors else "fail", len(long_anchors), "Slow levels must retain three-to-five-year causal percentiles.")
    add(rows, "daily_short_level_changes", "pass" if deltas else "fail", len(deltas), "Slow levels must also expose a standardized 20-day change for timing.")
    add(rows, "daily_multiscale_finite", "pass" if np.isfinite(transformed.to_numpy(dtype=float)).all() else "fail", len(transformed_cols), "Multiscale transformed inputs must be finite.")
    add(rows, "daily_availability_flags", "pass" if any(c.endswith("__available") for c in daily_cols) else "warn", sum(c.endswith("__available") for c in daily_cols), "Availability flags distinguish not-yet-observable data from neutral standardized values.")

    metadata_path = ROOT / daily_cfg["outputs"]["metadata"]
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        final_cols = list(metadata.get("feature_columns", []))
        required = list(daily_cfg["data"].get("required_panel_features", []))
        missing_required = [name for name in required if not any(c == name or c.startswith(f"{name}__") for c in final_cols)]
        add(rows, "daily_required_economic_anchors", "pass" if not missing_required else "fail", len(required) - len(missing_required), f"missing={missing_required}")

    dataset_check(ROOT / daily_cfg["outputs"]["dataset"], rows)
    dataset_check(ROOT / monthly_cfg["outputs"]["asset_dataset"], rows)

    report = pd.DataFrame(rows)
    out_csv = ROOT / "output" / "reports" / "point_in_time_pipeline_audit.csv"
    out_md = ROOT / "output" / "reports" / "point_in_time_pipeline_audit.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# Point-in-Time Pipeline Audit",
        "",
        "This audit checks causal filling, rolling normalization, release-date alignment and label ordering.",
        "",
        report.to_markdown(index=False),
        "",
        "Timing policy: `config/feature_timing_policy.yaml`.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(out_csv.relative_to(ROOT)), "md": str(out_md.relative_to(ROOT)), "checks": len(report), "failed": int(report["status"].eq("fail").sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
