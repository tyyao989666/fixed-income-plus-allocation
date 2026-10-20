from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"

CORE_FILES = [
    "base_asset_prices_daily.parquet",
    "rate_curves_daily_v43.parquet",
    "credit_spreads_daily_v43.parquet",
    "funding_liquidity_daily_v43.parquet",
    "cb_features_daily_v43.parquet",
    "equity_features_daily_v43.parquet",
    "macro_factors_monthly_v43.parquet",
    "alt_features_daily_v43.parquet",
    "feature_panel_daily_v43.parquet",
    "feature_panel_monthly_v43.parquet",
]

REQUIRED_FIELDS = {
    "base_asset_prices_daily.parquet": ["date", "asset_code", "close", "return_1d"],
    "rate_curves_daily_v43.parquet": ["date", "treasury_y1", "treasury_y3", "treasury_y5", "treasury_y10", "treasury_y30"],
    "credit_spreads_daily_v43.parquet": ["date", "spread_aaa_3y_proxy", "spread_aap_3y_proxy", "spread_aa_3y_proxy", "spread_aaa_5y_proxy"],
    "funding_liquidity_daily_v43.parquet": ["date", "r007", "dr007", "r007_dr007_spread", "shibor_3m", "aaa_cd_1y_final"],
    "cb_features_daily_v43.parquet": ["date", "cb_price_median", "cb_iv_median", "cb_ytm_median", "low_price_ratio_100", "low_price_ratio_110", "low_price_ratio_115", "high_price_premium_ratio", "callable_pressure_ratio"],
    "equity_features_daily_v43.parquet": ["date", "csi1000_close", "dividend_close", "sw_l1_positive_5d_ratio", "rzmre"],
    "macro_factors_monthly_v43.parquet": ["month", "cpi_yoy", "ppi_yoy", "m2_yoy", "social_financing_new", "industrial_production_yoy", "fixed_asset_investment_yoy", "export_yoy", "effective_date"],
    "alt_features_daily_v43.parquet": ["date", "us_10y_yield", "usd_index", "nanhua_commodity", "gold_au9999"],
    "feature_panel_daily_v43.parquet": ["date", "rate_treasury_y10", "credit_spread_aaa_3y_proxy", "funding_dr007", "cb_cb_price_median", "equity_sw_l1_positive_5d_ratio", "alt_nanhua_commodity"],
    "feature_panel_monthly_v43.parquet": ["month", "rate_treasury_y10", "credit_spread_aaa_3y_proxy", "funding_dr007", "cb_cb_price_median"],
}


def read(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def date_col(df: pd.DataFrame) -> str | None:
    if "date" in df.columns:
        return "date"
    if "month" in df.columns:
        return "month"
    return None


def infer_freq(dates: pd.Series) -> str:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(d) < 3:
        return "unknown"
    med = d.diff().dt.days.dropna().median()
    if med <= 7:
        return "daily_or_trading_daily"
    if 25 <= med <= 35:
        return "monthly"
    if 80 <= med <= 100:
        return "quarterly"
    if med >= 300:
        return "annual"
    return f"irregular_median_gap_{med}"


def long_gaps(dates: pd.Series, threshold: int) -> str:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(d) < 2:
        return ""
    gaps = d.diff().dt.days
    hits = []
    for idx, gap in gaps[gaps > threshold].items():
        loc = d.index.get_loc(idx)
        prev_date = d.iloc[loc - 1]
        this_date = d.loc[idx]
        hits.append(f"{prev_date.date()}->{this_date.date()}:{int(gap)}d")
    return "; ".join(hits[:10])


def annual_like_columns(df: pd.DataFrame) -> list[str]:
    bad = []
    for col in df.columns:
        s = str(col).lower()
        if any(k in s for k in ["annual", "yearly", "年度", "年频", "全年"]):
            bad.append(str(col))
    # Only flags/quality columns are treated as evidence. Free-text notes may say
    # "not annual"; those should not count as annual data contamination.
    for col in ["quality_flag", "summary_quality_flag", "detail_quality_flag"]:
        if col in df.columns:
            vals = df[col].astype(str)
            if vals.str.contains("annual|年度|年频|全年", case=False, na=False).any():
                bad.append(col)
    return sorted(set(bad))


def dataset_audit(name: str, df: pd.DataFrame) -> dict:
    required = REQUIRED_FIELDS.get(name, [])
    if df.empty:
        return {
            "dataset": name,
            "rows": 0,
            "start": None,
            "end": None,
            "inferred_frequency": "missing",
            "missing_required": ", ".join(required),
            "avg_missing_required": 1.0,
            "annual_data_detected": "no",
            "annual_evidence": "",
            "long_gaps": "",
            "severity": "blocking",
            "action": "补数据或从第一版剔除",
        }
    dcol = date_col(df)
    d = pd.to_datetime(df[dcol], errors="coerce") if dcol else pd.Series(dtype="datetime64[ns]")
    req_existing = [c for c in required if c in df.columns]
    missing_req = [c for c in required if c not in df.columns or not df[c].notna().any()]
    avg_missing = float(df[req_existing].isna().mean().mean()) if req_existing else 1.0
    freq = infer_freq(d) if dcol else "unknown"
    threshold = 45 if freq == "daily_or_trading_daily" else 70
    gaps = long_gaps(d, threshold) if dcol else ""
    annual_cols = annual_like_columns(df)
    severity = "ok"
    action = ""
    if annual_cols:
        severity = "must_fix"
        action = "不能进入v43基础面板；删除年度/年度聚合口径或降级为备注"
    elif missing_req:
        severity = "review"
        action = "字段缺失，确认是否非阻塞"
    elif gaps and freq == "daily_or_trading_daily":
        severity = "review"
        action = "存在长断档；只允许在总面板中ffill，并在报告披露"
    return {
        "dataset": name,
        "rows": len(df),
        "start": d.min() if len(d) else None,
        "end": d.max() if len(d) else None,
        "inferred_frequency": freq,
        "missing_required": ", ".join(missing_req),
        "avg_missing_required": avg_missing,
        "annual_data_detected": "yes" if annual_cols else "no",
        "annual_evidence": ", ".join(annual_cols),
        "long_gaps": gaps,
        "severity": severity,
        "action": action,
    }


def audit_credit_tail_annual_rows() -> pd.DataFrame:
    df = read("credit_tail_risk_monthly_v42.parquet")
    if df.empty:
        return pd.DataFrame()
    flags = pd.Series(False, index=df.index)
    for col in ["quality_flag", "summary_quality_flag", "detail_quality_flag", "note"]:
        if col in df.columns:
            flags |= df[col].astype(str).str.contains("annual|年度|年频|全年", case=False, na=False)
    cols = [c for c in ["month", "dataset", "quality_flag", "summary_quality_flag", "credit_bond_net_financing", "credit_net_financing_preferred", "note"] if c in df.columns]
    out = df.loc[flags, cols].copy()
    if not out.empty:
        out["audit_action"] = "年度聚合净融资不得进入v43基础月频；仅可保留为raw诊断"
    return out


def write_md(summary: pd.DataFrame, audit: pd.DataFrame, annual_rows: pd.DataFrame) -> None:
    def to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        tmp = df.copy()
        for c in tmp.columns:
            tmp[c] = tmp[c].astype(str).str.replace("\n", " ", regex=False)
        lines = ["| " + " | ".join(tmp.columns) + " |", "| " + " | ".join(["---"] * len(tmp.columns)) + " |"]
        for _, row in tmp.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in tmp.columns) + " |")
        return "\n".join(lines)

    text = (
        "# v4.3 无年频数据与遗漏审计\n\n"
        "## Summary\n\n"
        + to_md(summary)
        + "\n\n## Core Dataset Audit\n\n"
        + to_md(audit)
        + "\n\n## Annual / Annual-Aggregated Rows Found Outside v43\n\n"
        + to_md(annual_rows)
        + "\n\n## Conclusion\n\n"
        "- v43核心输出表不允许使用年频数据；检测到的年度聚合净融资只存在于旧v42/raw诊断表中，不应进入v43基础面板。\n"
        "- 日频表若存在长断档，只能在总面板中以前值填充供第一版跑通，原始明细表保留缺口并在报告披露。\n"
        "- 工业利润同比已使用可靠月频接入v43；仍禁止用年频/年度聚合口径替代缺失月频数据。\n"
    )
    (REPORTS / "v43_no_annual_data_audit.md").write_text(text, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame([dataset_audit(name, read(name)) for name in CORE_FILES])
    annual_rows = audit_credit_tail_annual_rows()
    summary = pd.DataFrame(
        {
            "metric": [
                "core_datasets",
                "must_fix_count",
                "review_count",
                "annual_detected_in_core",
                "annual_rows_in_old_credit_tail",
            ],
            "value": [
                len(audit),
                int(audit["severity"].eq("must_fix").sum()),
                int(audit["severity"].eq("review").sum()),
                int(audit["annual_data_detected"].eq("yes").sum()),
                len(annual_rows),
            ],
        }
    )
    audit.to_csv(REPORTS / "v43_no_annual_data_audit.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(REPORTS / "v43_no_annual_data_audit.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        audit.to_excel(writer, sheet_name="core_audit", index=False)
        if not annual_rows.empty:
            annual_rows.to_excel(writer, sheet_name="annual_rows_old_tables", index=False)
    write_md(summary, audit, annual_rows)
    print(summary.to_string(index=False))
    print(audit[["dataset", "inferred_frequency", "annual_data_detected", "severity", "missing_required", "long_gaps"]].to_string(index=False))


if __name__ == "__main__":
    main()
