from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
ASSET_PATH = DATA / "asset_returns_daily.parquet"
FEATURE_PATH = DATA / "feature_panel_daily_v43.parquet"
DATASET_PATH = DATA / "hierarchical_asset_itransformer_v14_residual_utility_dataset.npz"
META_PATH = REPORTS / "hierarchical_asset_itransformer_v14_residual_utility_metadata.json"


STRUCTURAL_NOTES = {
    "rate_30y_curve_proxy": "中债10年以上国债财富指数代理30Y，不是纯30Y可交易产品。",
    "equity_tech_growth": "科创50底层指数于2020-07-23发布，发布前不回填。",
    "alt_nonferrous": "南华有色指数于2018-06-29发布，发布前不回填。",
    "alt_reits": "公募REITs市场自2021年起形成，发布前不回填。",
    "alt_us_bond": "美国10Y国债收益率构造的曲线总收益代理，不是ETF成交价。",
}


def pct(value: float) -> str:
    return f"{value:.2%}"


def asset_coverage() -> pd.DataFrame:
    data = pd.read_parquet(ASSET_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "asset_code"]).sort_values(["asset_code", "date"])
    reference = pd.DatetimeIndex(
        data.loc[data["asset_code"].eq("equity_broad"), "date"].drop_duplicates().sort_values()
    )
    latest = data["date"].max()
    rows: list[dict[str, object]] = []
    for asset, frame in data.groupby("asset_code", sort=True):
        start, end = frame["date"].min(), frame["date"].max()
        expected = reference[(reference >= start) & (reference <= min(end, reference.max()))]
        observed = frame.loc[frame["close"].notna(), "date"].nunique()
        source = "; ".join(sorted(frame["data_source"].dropna().astype(str).unique()))
        missing_days = max(len(expected) - observed, 0)
        note = STRUCTURAL_NOTES.get(asset, "")
        status = "完整"
        if note:
            status = "结构性短历史/代理"
        if (latest - end).days > 5:
            status = "需关注滞后"
        rows.append(
            {
                "asset_code": asset,
                "asset_name": frame["asset_name"].dropna().iloc[-1] if frame["asset_name"].notna().any() else asset,
                "module": frame["module"].dropna().iloc[-1] if frame["module"].notna().any() else "",
                "start": start.date(),
                "end": end.date(),
                "rows": len(frame),
                "expected_cn_trade_days_since_start": len(expected),
                "close_coverage": observed / len(expected) if len(expected) else np.nan,
                "missing_cn_trade_days": missing_days,
                "ohlc_coverage": frame[["open", "high", "low", "close"]].notna().all(axis=1).mean(),
                "volume_coverage": frame["vol"].notna().mean(),
                "source": source,
                "status": status,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def feature_coverage() -> pd.DataFrame:
    data = pd.read_parquet(FEATURE_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    excluded = {"date", "source", "quality_flag", "frequency", "is_proxy", "note"}
    rows = []
    for column in data.columns:
        if column in excluded:
            continue
        valid = data[column].notna()
        rows.append(
            {
                "feature": column,
                "start": data.loc[valid, "date"].min().date() if valid.any() else None,
                "end": data.loc[valid, "date"].max().date() if valid.any() else None,
                "coverage": valid.mean(),
                "missing_rows": int((~valid).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage", "feature"], ascending=[True, True])


def target_coverage() -> pd.DataFrame:
    arrays = np.load(DATASET_PATH, allow_pickle=True)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    dates = pd.to_datetime(arrays["dates"], unit="ns", origin="unix")
    rows = []
    for key, label in [("asset_target_mask", "asset"), ("sleeve_target_mask", "sleeve")]:
        mask = arrays[key]
        for index, horizon in enumerate(metadata["horizons_days"]):
            rows.append(
                {
                    "target_level": label,
                    "horizon_days": horizon,
                    "valid_cells": int(mask[:, index].sum()),
                    "total_cells": int(mask[:, index].size),
                    "coverage": float(mask[:, index].mean()),
                    "last_valid_as_of": dates[np.flatnonzero(mask[:, index].any(axis=1))[-1]].date(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    assets = asset_coverage()
    features = feature_coverage()
    targets = target_coverage()
    asset_csv = REPORTS / "model_asset_data_coverage_20260721.csv"
    feature_csv = REPORTS / "model_feature_data_coverage_20260721.csv"
    target_csv = REPORTS / "model_target_coverage_20260721.csv"
    assets.to_csv(asset_csv, index=False, encoding="utf-8-sig")
    features.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    targets.to_csv(target_csv, index=False, encoding="utf-8-sig")

    lagged = assets.loc[assets["status"].eq("需关注滞后")]
    structural = assets.loc[assets["status"].eq("结构性短历史/代理")]
    lines = [
        "# 模型数据补全与时点审计",
        "",
        f"- 资产数：{len(assets)}",
        f"- 资产日频总行数：{int(assets['rows'].sum())}",
        f"- 资产总日期：{assets['start'].min()} 至 {assets['end'].max()}",
        f"- 状态特征日期：{pd.read_parquet(FEATURE_PATH)['date'].min()} 至 {pd.read_parquet(FEATURE_PATH)['date'].max()}",
        f"- 需关注的资产级滞后：{len(lagged)}",
        f"- 合法的结构性短历史或代理：{len(structural)}",
        "- 2026年7月尚未结束，月频数据只写到2026-06-30。",
        "- 未上市、未发布的数据没有事后回填；收益标签从as-of次日开始，尾部不足完整未来区间时不生成标签。",
        "",
        "## 结构性说明",
        "",
    ]
    for row in structural.itertuples(index=False):
        lines.append(f"- {row.asset_code}（{row.asset_name}）：{row.note}")
    lines.extend(["", "## 标签可用率", ""])
    for row in targets.itertuples(index=False):
        lines.append(
            f"- {row.target_level} {row.horizon_days}日：{pct(row.coverage)}，最后可验证as-of为{row.last_valid_as_of}。"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- `{asset_csv.relative_to(ROOT)}`：逐资产起止、行情字段覆盖、数据源与代理说明。",
            f"- `{feature_csv.relative_to(ROOT)}`：逐特征起止与缺失率。",
            f"- `{target_csv.relative_to(ROOT)}`：5/20/60日监督标签覆盖率。",
        ]
    )
    report = REPORTS / "data_completion_status_20260721.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"assets": len(assets), "asset_rows": int(assets["rows"].sum()), "lagged": len(lagged), "report": str(report.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
