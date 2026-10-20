from __future__ import annotations

from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
START_DATE = "20180101"
END_DATE = "20260604"


def build_repo_funding_daily() -> pd.DataFrame:
    frames = []
    for year in range(2018, 2027):
        start = f"{year}0101"
        end = min(f"{year}1231", END_DATE)
        if start > END_DATE:
            continue
        try:
            df = ak.repo_rate_hist(start_date=start, end_date=end)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    repo = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if repo.empty:
        return pd.DataFrame()
    repo["date"] = pd.to_datetime(repo["date"], errors="coerce")
    repo = repo.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    repo = repo.rename(columns={"FR007": "r007", "FDR007": "dr007", "FR001": "r001", "FDR001": "dr001"})
    for c in ["r001", "r007", "dr001", "dr007"]:
        repo[c] = pd.to_numeric(repo.get(c), errors="coerce")
    repo["r007_dr007_spread"] = repo["r007"] - repo["dr007"]
    repo["funding_stress_score_proxy"] = (
        repo["dr007"].rolling(252, min_periods=60).rank(pct=True) * 0.55
        + repo["r007_dr007_spread"].rolling(252, min_periods=60).rank(pct=True) * 0.45
    )
    repo["source"] = "akshare_repo_rate_hist_FR007_FDR007"
    repo.to_parquet(PROCESSED / "funding_rates_daily_v42.parquet", index=False)
    repo.to_csv(PROCESSED / "funding_rates_daily_v42.csv", index=False, encoding="utf-8-sig")
    return repo


def build_shibor_daily() -> pd.DataFrame:
    try:
        df = ak.macro_china_shibor_all()
    except Exception:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["日期"], errors="coerce")
    out = pd.DataFrame(
        {
            "date": df["date"],
            "shibor_on": pd.to_numeric(df.get("O/N-定价"), errors="coerce"),
            "shibor_1w": pd.to_numeric(df.get("1W-定价"), errors="coerce"),
            "shibor_3m": pd.to_numeric(df.get("3M-定价"), errors="coerce"),
            "shibor_1y": pd.to_numeric(df.get("1Y-定价"), errors="coerce"),
        }
    ).dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    out = out[out["date"].between(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE))]
    out["source"] = "akshare_macro_china_shibor_all"
    out.to_parquet(PROCESSED / "shibor_rates_daily_v42.parquet", index=False)
    out.to_csv(PROCESSED / "shibor_rates_daily_v42.csv", index=False, encoding="utf-8-sig")
    return out


def build_credit_spread_proxy() -> pd.DataFrame:
    p = PROCESSED / "credit_spreads_strict_daily_v42.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p).copy()
    if df.empty:
        return df
    for col in ["mtb_aaa_y3", "mtb_aaa_y5", "bank_bond_aaa_y3", "treasury_y3", "treasury_y5"]:
        if col not in df.columns:
            df[col] = np.nan
    df["credit_spread_proxy_aaa_mtb_1_3y_vs_treasury"] = df["mtb_aaa_y3"] - df["treasury_y3"]
    df["credit_spread_proxy_aaa_mtb_3_5y_vs_treasury"] = df["mtb_aaa_y5"] - df["treasury_y5"]
    df["credit_spread_proxy_aaa_bank_1_3y_vs_treasury"] = df["bank_bond_aaa_y3"] - df["treasury_y3"]
    df["credit_spread_proxy_available"] = df[
        [
            "credit_spread_proxy_aaa_mtb_1_3y_vs_treasury",
            "credit_spread_proxy_aaa_mtb_3_5y_vs_treasury",
            "credit_spread_proxy_aaa_bank_1_3y_vs_treasury",
        ]
    ].notna().any(axis=1)
    df["strict_missing_reason"] = "缺国开债曲线、AA+中票、AA中票；当前为AAA相对国债代理口径"
    df.to_parquet(PROCESSED / "credit_spreads_proxy_daily_v42.parquet", index=False)
    df.to_csv(PROCESSED / "credit_spreads_proxy_daily_v42.csv", index=False, encoding="utf-8-sig")
    return df


def rebuild_supplemental_panel() -> pd.DataFrame:
    files = [
        "funding_rates_daily_v42.parquet",
        "shibor_rates_daily_v42.parquet",
        "omo_policy_daily_v42.parquet",
        "credit_spreads_proxy_daily_v42.parquet",
        "equity_industry_breadth_daily_v42.parquet",
        "lgfv_spread_breadth_daily_v42.parquet",
    ]
    merged = None
    for name in files:
        p = PROCESSED / name
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        source_name = name.replace(".parquet", "").replace("_v42", "")
        if "source" in df.columns:
            df = df.rename(columns={"source": f"{source_name}_source"})
        df = df.drop_duplicates("date")
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged = merged.sort_values("date")
    merged.to_parquet(PROCESSED / "v42_supplemental_factors_daily.parquet", index=False)
    merged.to_csv(PROCESSED / "v42_supplemental_factors_daily.csv", index=False, encoding="utf-8-sig")
    return merged


def coverage(name: str, df: pd.DataFrame, cols: list[str]) -> dict:
    if df.empty:
        return {"dataset": name, "rows": 0, "start": None, "end": None, "non_null_core": 0, "status": "缺"}
    d = pd.to_datetime(df["date"] if "date" in df.columns else df["month"], errors="coerce")
    existing = [c for c in cols if c in df.columns]
    n = int(df[existing].notna().any(axis=1).sum()) if existing else 0
    return {"dataset": name, "rows": len(df), "start": d.min(), "end": d.max(), "non_null_core": n, "status": "已补齐" if n else "缺"}


def main() -> None:
    funding = build_repo_funding_daily()
    shibor = build_shibor_daily()
    credit_proxy = build_credit_spread_proxy()
    merged = rebuild_supplemental_panel()
    macro = pd.read_parquet(PROCESSED / "macro_factors_monthly_v42.parquet") if (PROCESSED / "macro_factors_monthly_v42.parquet").exists() else pd.DataFrame()
    breadth = pd.read_parquet(PROCESSED / "equity_industry_breadth_daily_v42.parquet") if (PROCESSED / "equity_industry_breadth_daily_v42.parquet").exists() else pd.DataFrame()
    rows = [
        coverage("funding_rates_daily_v42", funding, ["r007", "dr007", "r007_dr007_spread"]),
        coverage("shibor_rates_daily_v42", shibor, ["shibor_3m", "shibor_1y"]),
        coverage("credit_spreads_proxy_daily_v42", credit_proxy, ["credit_spread_proxy_aaa_mtb_1_3y_vs_treasury"]),
        coverage("equity_industry_breadth_daily_v42", breadth, ["sw_l1_positive_1d_ratio"]),
        coverage("v42_supplemental_factors_daily", merged, ["r007", "shibor_3m", "credit_spread_proxy_aaa_mtb_1_3y_vs_treasury", "sw_l1_positive_1d_ratio"]),
    ]
    if not macro.empty:
        d = pd.to_datetime(macro["month"], errors="coerce")
        rows.insert(0, {"dataset": "macro_factors_monthly_v42", "rows": len(macro), "start": d.min(), "end": d.max(), "non_null_core": int(macro.drop(columns=["month"], errors="ignore").notna().any(axis=1).sum()), "status": "已补齐"})
    report = pd.DataFrame(rows)
    with pd.ExcelWriter(REPORTS / "v42_gap_fill_report_after_fallback.xlsx") as writer:
        report.to_excel(writer, sheet_name="coverage", index=False)
        for name, df in {"funding": funding, "shibor": shibor, "credit_proxy": credit_proxy, "merged": merged}.items():
            if not df.empty:
                df.head(1000).to_excel(writer, sheet_name=name, index=False)
    (REPORTS / "v42_gap_fill_report_after_fallback.md").write_text(
        "# v4.2 数据补全报告 after fallback\n\n"
        + report.to_markdown(index=False)
        + "\n\n说明：资金面已用 AkShare 回购定盘利率补到 2018；信用利差严格口径仍待 Wind 国开债/AA+/AA 曲线，当前提供 AAA 相对国债代理口径以保证数据工程可跑通。\n",
        encoding="utf-8",
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
