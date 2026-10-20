from __future__ import annotations

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
CB_DB = Path(r"C:\Users\lenovo\Desktop\可转债数据库生成\output\20260605\转债数据库_20260605.xlsx")


def read_parquet(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def as_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=[col]).sort_values(col)
    return df


def add_meta(df: pd.DataFrame, source: str, quality_flag: str, frequency: str, is_proxy: bool) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source
    df["quality_flag"] = quality_flag
    df["frequency"] = frequency
    df["is_proxy"] = is_proxy
    return df


def drop_all_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that contain no usable observations in the final clean tables."""
    if df.empty:
        return df
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    return df.drop(columns=empty_cols) if empty_cols else df


def date_column_for(df: pd.DataFrame, frequency: str | None = None) -> str | None:
    if frequency == "monthly" and "month" in df.columns:
        return "month"
    if "date" in df.columns:
        return "date"
    if "month" in df.columns:
        return "month"
    return None


def common_date_window(dfs: list[pd.DataFrame], frequency: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for df in dfs:
        dcol = date_column_for(df, frequency)
        if df.empty or dcol is None:
            continue
        dates = pd.to_datetime(df[dcol], errors="coerce").dropna()
        if dates.empty:
            continue
        starts.append(dates.min())
        ends.append(dates.max())
    if not starts or not ends:
        raise ValueError("No valid date columns found for common window alignment.")
    return max(starts), min(ends)


def trim_to_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, frequency: str | None = None) -> pd.DataFrame:
    dcol = date_column_for(df, frequency)
    if df.empty or dcol is None:
        return df
    out = df.copy()
    out[dcol] = pd.to_datetime(out[dcol], errors="coerce")
    out = out[out[dcol].between(start, end)].sort_values(dcol).reset_index(drop=True)
    return out


def align_and_save_daily_outputs(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # A slow-moving source must not truncate every faster daily source.  Keep a
    # common research start, preserve each source's own latest observation, and
    # let the feature panel carry published values forward on the master trading
    # calendar.  Source freshness is audited separately.
    start, _ = common_date_window(list(outputs.values()), "daily")
    aligned: dict[str, pd.DataFrame] = {}
    for name, df in outputs.items():
        dcol = date_column_for(df, "daily")
        if df.empty or dcol is None:
            out = df
        else:
            dates = pd.to_datetime(df[dcol], errors="coerce")
            out = trim_to_window(df, start, dates.max(), "daily")
        out.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        aligned[name] = out
    return aligned


def align_and_save_monthly_outputs(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Macro releases can lag prices by several weeks.  Preserve the different
    # latest dates instead of deleting already-completed asset months.
    start, _ = common_date_window(list(outputs.values()), "monthly")
    aligned: dict[str, pd.DataFrame] = {}
    for name, df in outputs.items():
        dcol = date_column_for(df, "monthly")
        if df.empty or dcol is None:
            out = df
        else:
            dates = pd.to_datetime(df[dcol], errors="coerce")
            out = trim_to_window(df, start, dates.max(), "monthly")
        if "date" in out.columns and "month" in out.columns:
            out["date"] = out["month"]
        out.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        aligned[name] = out
    return aligned


def last_completed_month_end(as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the last calendar month that is fully observable today."""
    today = pd.Timestamp(date.today())
    reference = pd.Timestamp(as_of) if as_of is not None else today
    current_month_end = today + pd.offsets.MonthEnd(0)
    if reference.to_period("M") < today.to_period("M"):
        return reference + pd.offsets.MonthEnd(0)
    if today.normalize() >= current_month_end.normalize():
        return current_month_end.normalize()
    return (today.to_period("M") - 1).to_timestamp("M")


def align_return_tables(daily_start: pd.Timestamp, daily_end: pd.Timestamp, monthly_start: pd.Timestamp, monthly_end: pd.Timestamp) -> None:
    daily = read_parquet("asset_returns_daily.parquet")
    if not daily.empty:
        daily = trim_to_window(daily, daily_start, daily_end, "daily")
        daily.to_parquet(PROCESSED / "asset_returns_daily.parquet", index=False)
        monthly_rows = []
        for _, g in daily.sort_values(["asset_code", "date"]).groupby("asset_code", dropna=False):
            m = g.set_index("date").resample("ME").last().dropna(subset=["close"]).reset_index()
            if m.empty:
                continue
            m["return_1m"] = m["close"].pct_change()
            for col in ["asset_name", "asset_class", "module", "sub_module", "data_source", "fetch_time"]:
                if col in g.columns:
                    m[col] = g[col].dropna().iloc[-1] if g[col].notna().any() else np.nan
            asset_code = g["asset_code"].dropna().iloc[0] if g["asset_code"].notna().any() else np.nan
            m["asset_code"] = asset_code
            m["code"] = asset_code
            monthly_rows.append(m)
        if monthly_rows:
            monthly = pd.concat(monthly_rows, ignore_index=True, sort=False)
            monthly = trim_to_window(monthly, monthly_start, monthly_end, "daily")
            keep = [
                "date",
                "asset_code",
                "code",
                "asset_name",
                "asset_class",
                "module",
                "sub_module",
                "close",
                "return_1m",
                "data_source",
                "fetch_time",
            ]
            monthly[[c for c in keep if c in monthly.columns]].to_parquet(PROCESSED / "asset_returns_monthly.parquet", index=False)


def non_meta_cols(df: pd.DataFrame) -> list[str]:
    meta = {"date", "month", "effective_date", "source", "quality_flag", "frequency", "is_proxy", "note"}
    return [c for c in df.columns if c not in meta]


def completion_row(name: str, df: pd.DataFrame, required: list[str], blocker: bool = True) -> dict:
    if df.empty:
        return {
            "dataset": name,
            "rows": 0,
            "start": None,
            "end": None,
            "required_fields": ", ".join(required),
            "available_required": "",
            "missing_required": ", ".join(required),
            "missing_rate_core": 1.0,
            "quality_flag": "missing",
            "frequency": None,
            "is_proxy": None,
            "first_version_blocker": "yes" if blocker else "no",
            "note": "dataset missing",
        }
    available = [c for c in required if c in df.columns and df[c].notna().any()]
    missing = [c for c in required if c not in available]
    core = [c for c in required if c in df.columns]
    missing_rate = float(df[core].isna().mean().mean()) if core else 1.0
    quality = ", ".join(df["quality_flag"].dropna().astype(str).drop_duplicates().tolist()) if "quality_flag" in df.columns and len(df) else ""
    freq = ", ".join(df["frequency"].dropna().astype(str).drop_duplicates().tolist()) if "frequency" in df.columns and len(df) else ""
    dcol = date_column_for(df, freq)
    d = pd.to_datetime(df[dcol], errors="coerce") if dcol else pd.Series(dtype="datetime64[ns]")
    proxy = bool(df["is_proxy"].any()) if "is_proxy" in df.columns else False
    return {
        "dataset": name,
        "rows": len(df),
        "start": d.min() if len(d) else None,
        "end": d.max() if len(d) else None,
        "required_fields": ", ".join(required),
        "available_required": ", ".join(available),
        "missing_required": ", ".join(missing),
        "missing_rate_core": missing_rate,
        "quality_flag": quality,
        "frequency": freq,
        "is_proxy": proxy,
        "first_version_blocker": "no" if not missing or not blocker else "yes",
        "note": "",
    }


def build_base_asset_prices() -> pd.DataFrame:
    df = as_date(read_parquet("asset_returns_daily.parquet"))
    if df.empty:
        return df
    keep = [
        "date",
        "asset_code",
        "code",
        "asset_name",
        "asset_class",
        "module",
        "sub_module",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "return_1d",
        "data_source",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = add_meta(out, "asset_returns_daily.parquet", "direct_daily", "daily", False)
    out.to_parquet(PROCESSED / "base_asset_prices_daily.parquet", index=False)
    return out


def build_rate_curves() -> pd.DataFrame:
    raw = read_parquet("raw_chinabond_yield_curves.parquet")
    if raw.empty:
        raw = read_parquet("credit_yield_curves_daily_v42_long.parquet")
    if raw.empty:
        return raw
    raw = raw.copy()
    if "日期" in raw.columns:
        raw["date"] = pd.to_datetime(raw["日期"], errors="coerce")
        treasury = raw[raw["曲线名称"].astype(str).str.contains("国债收益率曲线", na=False)].copy()
        policy = raw[raw["曲线名称"].astype(str).str.contains("国开|政策性金融债", na=False)].copy()
        out = pd.DataFrame({"date": sorted(raw["date"].dropna().unique())})
        for src, prefix in [(treasury, "treasury"), (policy, "policy_bank")]:
            if src.empty:
                continue
            cols = {"1年": "y1", "3年": "y3", "5年": "y5", "10年": "y10", "30年": "y30"}
            tmp = src[["date"] + [c for c in cols if c in src.columns]].drop_duplicates("date")
            tmp = tmp.rename(columns={k: f"{prefix}_{v}" for k, v in cols.items() if k in tmp.columns})
            out = out.merge(tmp, on="date", how="left")
    else:
        long = as_date(raw)
        out = pd.DataFrame({"date": sorted(long["date"].dropna().unique())})
        for key, prefix in [("treasury", "treasury"), ("policy_bank", "policy_bank")]:
            tmp = long[long.get("curve_key", "").astype(str).eq(key)].copy()
            if tmp.empty:
                continue
            tmp = tmp.rename(columns={c: f"{prefix}_{c}" for c in ["y1", "y3", "y5", "y10", "y30"] if c in tmp.columns})
            out = out.merge(tmp[["date"] + [c for c in tmp.columns if c.startswith(prefix)]], on="date", how="left")
    out["treasury_10y_change_5d"] = out.get("treasury_y10", pd.Series(index=out.index, dtype=float)).diff(5)
    out["treasury_10y_change_20d"] = out.get("treasury_y10", pd.Series(index=out.index, dtype=float)).diff(20)
    if "treasury_y30" in out.columns and "treasury_y10" in out.columns:
        out["curve_30y_10y"] = out["treasury_y30"] - out["treasury_y10"]
    if "treasury_y10" in out.columns and "treasury_y1" in out.columns:
        out["curve_10y_1y"] = out["treasury_y10"] - out["treasury_y1"]
    has_policy = any(c.startswith("policy_bank") and out[c].notna().any() for c in out.columns)
    out = add_meta(
        out,
        "raw_chinabond_yield_curves.parquet",
        "direct_daily" if has_policy else "direct_daily_policy_bank_missing",
        "daily",
        False,
    )
    out.to_parquet(PROCESSED / "rate_curves_daily_v43.parquet", index=False)
    return out


def build_credit_spreads(rate: pd.DataFrame) -> pd.DataFrame:
    proxy = as_date(read_parquet("credit_spreads_proxy_daily_v42.parquet"))
    strict = as_date(read_parquet("credit_spreads_strict_daily_v42.parquet"))
    base = proxy if not proxy.empty else strict
    if base.empty:
        return base
    out = base[["date"]].drop_duplicates().copy()
    source_cols = [
        "credit_spread_aaa_1_3y",
        "credit_spread_aa_plus_1_3y",
        "credit_spread_aa_1_3y",
        "credit_spread_aaa_3_5y",
        "credit_spread_proxy_aaa_mtb_1_3y_vs_treasury",
        "credit_spread_proxy_aaa_mtb_3_5y_vs_treasury",
        "credit_spread_proxy_aaa_bank_1_3y_vs_treasury",
    ]
    for c in source_cols:
        if c in base.columns:
            out[c] = pd.to_numeric(base[c], errors="coerce")
    out["spread_aaa_3y"] = out.get("credit_spread_aaa_1_3y", pd.Series(index=out.index, dtype=float))
    out["spread_aap_3y"] = out.get("credit_spread_aa_plus_1_3y", pd.Series(index=out.index, dtype=float))
    out["spread_aa_3y"] = out.get("credit_spread_aa_1_3y", pd.Series(index=out.index, dtype=float))
    out["spread_aaa_5y"] = out.get("credit_spread_aaa_3_5y", pd.Series(index=out.index, dtype=float))
    out["spread_aaa_3y_proxy"] = out["spread_aaa_3y"].combine_first(out.get("credit_spread_proxy_aaa_mtb_1_3y_vs_treasury"))
    out["spread_aaa_5y_proxy"] = out["spread_aaa_5y"].combine_first(out.get("credit_spread_proxy_aaa_mtb_3_5y_vs_treasury"))
    out["spread_aap_3y_proxy"] = out["spread_aap_3y"].combine_first(out["spread_aaa_3y_proxy"])
    out["spread_aa_3y_proxy"] = out["spread_aa_3y"].combine_first(out["spread_aaa_3y_proxy"])
    out["spread_aa_aaa_proxy"] = out["spread_aa_3y_proxy"] - out["spread_aaa_3y_proxy"]
    for c in ["spread_aaa_3y_proxy", "spread_aap_3y_proxy", "spread_aa_3y_proxy", "spread_aaa_5y_proxy"]:
        out[f"{c}_change_5d"] = out[c].diff(5)
        out[f"{c}_change_20d"] = out[c].diff(20)
        out[f"{c}_pct_2018_to_t"] = out[c].expanding(min_periods=252).rank(pct=True)
    out["note"] = "严格AA+/AA/国开债利差未完整取得；第一版使用AAA中票/商业银行债相对国债代理补齐。"
    out = add_meta(out, "credit_spreads_proxy_daily_v42.parquet", "proxy_daily", "daily", True)
    out = drop_all_empty_columns(out)
    out.to_parquet(PROCESSED / "credit_spreads_daily_v43.parquet", index=False)
    return out


def build_funding_liquidity() -> pd.DataFrame:
    funding = as_date(read_parquet("funding_rates_daily_v42.parquet"))
    shibor = as_date(read_parquet("shibor_rates_daily_v42.parquet"))
    cd = as_date(read_parquet("aaa_cd_1y_proxy_daily.parquet"))
    omo = as_date(read_parquet("omo_policy_daily_v42.parquet"))
    if funding.empty and shibor.empty and cd.empty and omo.empty:
        return pd.DataFrame()
    dates = pd.concat([df[["date"]] for df in [funding, shibor, cd, omo] if not df.empty]).drop_duplicates()
    out = dates.sort_values("date")
    for df in [funding, shibor, cd, omo]:
        if df.empty:
            continue
        cols = [c for c in df.columns if c != "source"]
        out = out.merge(df[cols].drop_duplicates("date"), on="date", how="left")
    if "r007" in out.columns and "dr007" in out.columns:
        out["r007_dr007_spread"] = out["r007"] - out["dr007"]
    if "dr007" in out.columns:
        out["dr007_avg_20d"] = out["dr007"].rolling(20, min_periods=5).mean()
    if "shibor_3m" in out.columns:
        out["shibor_3m_change_20d"] = out["shibor_3m"].diff(20)
    if "aaa_cd_1y_final" in out.columns:
        out["aaa_cd_1y_change_20d"] = out["aaa_cd_1y_final"].diff(20)
    out["note"] = "AAA同业存单1Y为中债商业银行普通债AAA 1Y/Shibor回退代理；OMO仅覆盖2024-11以后。"
    out = add_meta(out, "funding_v42+shibor_v42+aaa_cd_proxy+omo_v42", "proxy_daily", "daily", True)
    out.to_parquet(PROCESSED / "funding_liquidity_daily_v43.parquet", index=False)
    return out


def build_cb_features_from_existing() -> pd.DataFrame:
    cb = as_date(read_parquet("existing_cb_market_features_daily.parquet"))
    if cb.empty:
        return cb
    rename = {
        "cb_price_median": "cb_price_median",
        "cb_iv_median": "cb_iv_median",
        "cb_ytm_median": "cb_ytm_median",
        "cb_premium_median": "cb_premium_median",
        "cb_amount_sum": "cb_turnover_amount",
        "cb_balance_sum": "cb_balance",
    }
    out = cb[["date"] + [c for c in rename if c in cb.columns]].rename(columns=rename).copy()
    if "low_price_cb_ratio" in cb.columns:
        out["low_price_ratio_100_proxy"] = cb["low_price_cb_ratio"]
    if "high_price_cb_ratio" in cb.columns:
        out["high_price_ratio_proxy"] = cb["high_price_cb_ratio"]
    out["source_detail"] = "existing_cb_market_features_daily.parquet"
    return out


def build_cb_feature_detail_from_xlsx(out: pd.DataFrame) -> pd.DataFrame:
    if not CB_DB.exists():
        out["note"] = "本地转债数据库未找到，低价细分/强赎压力只能使用既有代理列。"
        return out
    try:
        price = pd.read_excel(CB_DB, sheet_name="cb_转债收盘价")
        bal = pd.read_excel(CB_DB, sheet_name="cb_债券余额")
        premium = pd.read_excel(CB_DB, sheet_name="cb_平价溢价率")
        parity = pd.read_excel(CB_DB, sheet_name="cb_平价")
    except Exception as exc:
        out["note"] = f"读取本地转债数据库失败：{exc}"
        return out
    for df in [price, bal, premium, parity]:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    common_dates = sorted(set(price["日期"]).intersection(set(bal["日期"])))
    rows = []
    price_idx = price.set_index("日期")
    bal_idx = bal.set_index("日期")
    prem_idx = premium.set_index("日期") if not premium.empty else None
    parity_idx = parity.set_index("日期") if not parity.empty else None
    for d in common_dates:
        p = pd.to_numeric(price_idx.loc[d].drop(labels=["__index_level_0__"], errors="ignore"), errors="coerce")
        b = pd.to_numeric(bal_idx.loc[d].drop(labels=["__index_level_0__"], errors="ignore"), errors="coerce")
        b = b.where(b > 0)
        total = b.sum(skipna=True)
        row = {"date": d}
        if total and total > 0:
            row["low_price_ratio_100"] = b[p < 100].sum(skipna=True) / total
            row["low_price_ratio_110"] = b[p < 110].sum(skipna=True) / total
            row["low_price_ratio_115"] = b[p < 115].sum(skipna=True) / total
            row["cb_balance_detail"] = total
            if prem_idx is not None:
                prem = pd.to_numeric(prem_idx.loc[d].drop(labels=["__index_level_0__"], errors="ignore"), errors="coerce")
                row["high_price_premium_ratio"] = b[(p > 130) & (prem > 30)].sum(skipna=True) / total
            if parity_idx is not None:
                par = pd.to_numeric(parity_idx.loc[d].drop(labels=["__index_level_0__"], errors="ignore"), errors="coerce")
                row["callable_pressure_ratio"] = b[par >= 130].sum(skipna=True) / total
        rows.append(row)
    detail = pd.DataFrame(rows)
    if detail.empty:
        out["note"] = "本地转债数据库细分指标为空，低价细分/强赎压力只能使用既有代理列。"
        return out
    merged = out.merge(detail, on="date", how="left")
    merged["cb_balance"] = merged.get("cb_balance", pd.Series(index=merged.index, dtype=float)).combine_first(merged.get("cb_balance_detail"))
    merged["note"] = "低价100/110/115、高价高溢价、强赎压力来自本地20260605转债数据库；强赎压力用平价>=130余额占比作基础代理。"
    return merged.drop(columns=["cb_balance_detail"], errors="ignore")


def build_cb_features() -> pd.DataFrame:
    out = build_cb_features_from_existing()
    if out.empty:
        return out
    out = build_cb_feature_detail_from_xlsx(out)
    out = out[out["date"] >= pd.Timestamp("2018-01-01")].copy()
    out = add_meta(out, "existing_cb_market_features_daily+local_cb_database_20260605", "proxy_daily", "daily", True)
    out.to_parquet(PROCESSED / "cb_features_daily_v43.parquet", index=False)
    return out


def build_equity_features() -> pd.DataFrame:
    idx = as_date(read_parquet("index_valuation_filled_daily.parquet"))
    breadth = as_date(read_parquet("equity_industry_breadth_daily_v42.parquet"))
    margin = as_date(read_parquet("margin_trading_daily_v42.parquet"))
    asset = as_date(read_parquet("feature_panel_daily.parquet"))
    if idx.empty and asset.empty:
        return pd.DataFrame()
    dates = []
    for df in [idx, breadth, margin, asset]:
        if not df.empty and "date" in df.columns:
            dates.append(df[["date"]])
    out = pd.concat(dates).drop_duplicates().sort_values("date")
    if not idx.empty:
        for name in idx["index_name"].dropna().unique():
            tmp = idx[idx["index_name"] == name][["date", "close", "amount", "pe", "pb"]].copy()
            tmp = tmp.rename(columns={c: f"{name}_{c}" for c in ["close", "amount", "pe", "pb"]})
            out = out.merge(tmp, on="date", how="left")
    if not asset.empty:
        asset_map = {
            "equity_hs300": "hs300",
            "equity_a500": "csi_a500",
            "equity_all_a": "all_a",
            "equity_growth": "growth_proxy",
            "equity_dividend": "dividend_proxy",
        }
        for asset_code, prefix in asset_map.items():
            tmp = asset[asset["asset_code"].astype(str).eq(asset_code)][["date", "close", "amount", "return_1d"]].copy()
            if tmp.empty:
                continue
            tmp = tmp.rename(columns={c: f"{prefix}_{c}" for c in ["close", "amount", "return_1d"]})
            out = out.merge(tmp, on="date", how="left")
    for df in [breadth, margin]:
        if not df.empty:
            out = out.merge(df.drop(columns=["source"], errors="ignore").drop_duplicates("date"), on="date", how="left")
    out["note"] = "中证A500/万得全A/成长若无指数原始Wind字段，使用资产池ETF/代理口径。"
    out = add_meta(out, "index_valuation_filled_daily+feature_panel_daily+margin+industry_breadth", "proxy_daily", "daily", True)
    out = drop_all_empty_columns(out)
    out.to_parquet(PROCESSED / "equity_features_daily_v43.parquet", index=False)
    return out


def build_macro() -> pd.DataFrame:
    macro = read_parquet("macro_factors_monthly_v42.parquet")
    if macro.empty:
        return macro
    out = macro.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    if "effective_date_conservative" in out.columns:
        out["effective_date"] = pd.to_datetime(out["effective_date_conservative"], errors="coerce")
    else:
        out["effective_date"] = out["month"] + pd.offsets.Day(25)
    industrial = parse_wind_industrial_output_monthly()
    if not industrial.empty:
        out = out.drop(columns=["industrial_production_yoy"], errors="ignore").merge(industrial, on="month", how="left")
    profit = read_parquet("industrial_profit_monthly_v43.parquet")
    if not profit.empty:
        profit = profit[["month", "industrial_profit_yoy", "industrial_profit_accum_yoy"]].copy()
        profit["month"] = pd.to_datetime(profit["month"], errors="coerce")
        out = out.drop(columns=["industrial_profit_yoy", "industrial_profit_accum_yoy"], errors="ignore").merge(profit, on="month", how="left")
    if "industrial_profit_yoy" not in out.columns:
        out["industrial_profit_yoy"] = np.nan
    out["note"] = "月频宏观使用保守生效日；工业增加值来自Wind月频缓存；工业利润若存在则来自Wind月频，不使用年频替代。"
    out = add_meta(out.drop(columns=["effective_date_conservative"], errors="ignore"), "macro_factors_monthly_v42.parquet", "direct_monthly", "monthly", False)
    out.to_parquet(PROCESSED / "macro_factors_monthly_v43.parquet", index=False)
    return out


def parse_wind_industrial_output_monthly() -> pd.DataFrame:
    path = ROOT / "data_raw" / "wind" / "v43_macro_industrial_output_monthly_2018_2026.json"
    if not path.exists():
        return pd.DataFrame()
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    blocks = data.get("data", data) if isinstance(data, dict) else data
    frames: list[pd.DataFrame] = []
    if not isinstance(blocks, list):
        return pd.DataFrame()
    for block in blocks:
        rows = block.get("rows") if isinstance(block, dict) else None
        cols = [c.get("name", f"col_{i}") for i, c in enumerate(block.get("columns", []))] if isinstance(block, dict) else []
        if not rows or not cols:
            continue
        max_len = max(len(r) for r in rows)
        cols = (cols + [f"col_{i}" for i in range(len(cols), max_len)])[:max_len]
        frames.append(pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=cols))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    date_col = next((c for c in df.columns if "日期" in str(c)), None)
    if not date_col:
        return pd.DataFrame()
    value_cols = [c for c in df.columns if "工业增加值" in str(c) and "同比" in str(c)]
    if not value_cols:
        return pd.DataFrame()
    preferred = next((c for c in value_cols if "合并" in str(c)), value_cols[0])
    fallback = next((c for c in value_cols if c != preferred), preferred)
    out = pd.DataFrame({"month": pd.to_datetime(df[date_col], errors="coerce") + pd.offsets.MonthEnd(0)})
    out["industrial_production_yoy"] = pd.to_numeric(df[preferred], errors="coerce").combine_first(pd.to_numeric(df[fallback], errors="coerce"))
    out = out.dropna(subset=["month"]).drop_duplicates("month").sort_values("month")
    out["industrial_production_source"] = "wind_v43_macro_industrial_output_monthly"
    return out


def build_alt() -> pd.DataFrame:
    alt = as_date(read_parquet("alt_overseas_filled_daily.parquet"))
    if alt.empty:
        return alt
    out = alt.copy()
    out["real_rate_proxy"] = np.nan
    rate = read_parquet("rate_curves_daily_v43.parquet")
    macro = read_parquet("macro_factors_monthly_v43.parquet")
    if not rate.empty and not macro.empty and "treasury_y10" in rate.columns and "cpi_yoy" in macro.columns:
        visible_col = "effective_date" if "effective_date" in macro.columns else "month"
        m = macro[[visible_col, "cpi_yoy"]].dropna().copy()
        m["visible_date"] = pd.to_datetime(m[visible_col], errors="coerce")
        m = m.dropna(subset=["visible_date"]).sort_values("visible_date")
        tmp = pd.merge_asof(
            out[["date"]].sort_values("date"),
            m[["visible_date", "cpi_yoy"]],
            left_on="date",
            right_on="visible_date",
            direction="backward",
        )
        out = out.merge(rate[["date", "treasury_y10"]], on="date", how="left")
        out["real_rate_proxy"] = out["treasury_y10"] - tmp["cpi_yoy"].to_numpy()
    reits = read_parquet("reits_distribution_yield_v43.parquet")
    out["reits_index_close"] = pd.Series(np.nan, index=out.index, dtype="float64")
    out["reits_distribution_yield"] = pd.Series(np.nan, index=out.index, dtype="float64")
    if not reits.empty:
        reits = reits.copy()
        reits["date"] = pd.to_datetime(reits["date"], errors="coerce")
        if "reits_index_close" not in reits.columns:
            reits["reits_index_close"] = np.nan
        reits = reits.rename(columns={"reits_index_close": "reits_index_close_monthly", "reits_distribution_yield": "reits_distribution_yield_monthly"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce").astype("datetime64[ns]")
        reits["date"] = pd.to_datetime(reits["date"], errors="coerce").astype("datetime64[ns]")
        out = pd.merge_asof(
            out.sort_values("date"),
            reits[["date", "reits_index_close_monthly", "reits_distribution_yield_monthly"]].dropna(subset=["date"]).sort_values("date"),
            on="date",
            direction="backward",
        )
        out["reits_index_close"] = out["reits_index_close"].where(
            out["reits_index_close"].notna(),
            pd.to_numeric(out["reits_index_close_monthly"], errors="coerce"),
        )
        out["reits_distribution_yield"] = out["reits_distribution_yield"].where(
            out["reits_distribution_yield"].notna(),
            pd.to_numeric(out["reits_distribution_yield_monthly"], errors="coerce"),
        )
        out = out.drop(columns=["reits_index_close_monthly", "reits_distribution_yield_monthly"], errors="ignore")
    out["note"] = "黄金/南华商品/美元/美债10Y已按现有日频表；REITs分派率若存在则为Wind单只REITs月度均值代理。"
    out = add_meta(out, "alt_overseas_filled_daily.parquet", "proxy_daily", "daily", True)
    out = drop_all_empty_columns(out)
    out.to_parquet(PROCESSED / "alt_features_daily_v43.parquet", index=False)
    return out


def wide_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    keep = ["date"] + non_meta_cols(df)
    tmp = df[keep].copy()
    return tmp.rename(columns={c: f"{prefix}_{c}" for c in tmp.columns if c != "date"})


def build_feature_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = {
        "asset": read_parquet("base_asset_prices_daily.parquet"),
        "rate": read_parquet("rate_curves_daily_v43.parquet"),
        "credit": read_parquet("credit_spreads_daily_v43.parquet"),
        "funding": read_parquet("funding_liquidity_daily_v43.parquet"),
        "cb": read_parquet("cb_features_daily_v43.parquet"),
        "equity": read_parquet("equity_features_daily_v43.parquet"),
        "alt": read_parquet("alt_features_daily_v43.parquet"),
    }
    base_dates = parts["asset"][["date"]].drop_duplicates() if not parts["asset"].empty and "date" in parts["asset"].columns else pd.DataFrame()
    dates = [base_dates] if not base_dates.empty else [df[["date"]] for df in parts.values() if not df.empty and "date" in df.columns]
    if not dates:
        return pd.DataFrame(), pd.DataFrame()
    panel = pd.concat(dates).drop_duplicates().sort_values("date")
    for prefix, df in parts.items():
        if prefix == "asset":
            continue
        panel = panel.merge(wide_prefix(as_date(df), prefix), on="date", how="left")
    feature_cols = [c for c in panel.columns if c != "date"]
    panel[feature_cols] = panel[feature_cols].ffill()
    panel = panel.copy()
    panel["source"] = "v43_standardized_existing_data"
    panel["quality_flag"] = "mixed_direct_proxy_daily"
    panel["frequency"] = "daily"
    panel["is_proxy"] = True
    panel["note"] = "以基础资产交易日为主轴；状态类特征从各原始日频/月频表对齐后向前填充，原始明细表未被覆盖。"
    panel = drop_all_empty_columns(panel)
    panel.to_parquet(PROCESSED / "feature_panel_daily_v43.parquet", index=False)
    monthly = panel.copy()
    monthly["month"] = monthly["date"] + pd.offsets.MonthEnd(0)
    monthly = monthly.sort_values("date").groupby("month", as_index=False).last()
    monthly = monthly[monthly["month"].le(last_completed_month_end(panel["date"].max()))].copy()
    monthly["source"] = "feature_panel_daily_v43_month_end"
    monthly["quality_flag"] = "mixed_direct_proxy_monthly"
    monthly["frequency"] = "monthly"
    monthly["is_proxy"] = True
    monthly = drop_all_empty_columns(monthly)
    monthly.to_parquet(PROCESSED / "feature_panel_monthly_v43.parquet", index=False)
    return panel, monthly


def write_report(datasets: dict[str, tuple[pd.DataFrame, list[str], bool]]) -> pd.DataFrame:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = [completion_row(name, df, required, blocker) for name, (df, required, blocker) in datasets.items()]
    report = pd.DataFrame(rows)
    report["status_bucket"] = np.select(
        [
            report["quality_flag"].astype(str).str.contains("direct_daily"),
            report["quality_flag"].astype(str).str.contains("direct_monthly"),
            report["quality_flag"].astype(str).str.contains("proxy_daily"),
            report["quality_flag"].astype(str).str.contains("proxy_monthly"),
            report["quality_flag"].astype(str).str.contains("low_confidence"),
            report["first_version_blocker"].eq("no") & report["missing_required"].astype(str).ne(""),
        ],
        ["direct_daily", "direct_monthly", "proxy_daily", "proxy_monthly", "low_confidence", "missing_but_not_blocking"],
        default="needs_review",
    )
    blocking = report[report["first_version_blocker"].eq("yes")]
    summary = pd.DataFrame(
        {
            "metric": ["datasets", "blocking_datasets", "non_blocking_or_complete", "latest_daily_panel_date"],
            "value": [
                len(report),
                len(blocking),
                int(report["first_version_blocker"].eq("no").sum()),
                str(datasets["feature_panel_daily_v43"][0]["date"].max().date()) if not datasets["feature_panel_daily_v43"][0].empty else "",
            ],
        }
    )
    xlsx = REPORTS / "v43_basic_data_completion_report.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        report.to_excel(writer, sheet_name="coverage", index=False)
        for name, (df, _, _) in datasets.items():
            if not df.empty:
                df.head(1000).to_excel(writer, sheet_name=name[:31], index=False)
    def to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        text_df = df.copy()
        for col in text_df.columns:
            text_df[col] = text_df[col].astype(str).str.replace("\n", " ", regex=False)
        cols = list(text_df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in text_df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    md = REPORTS / "v43_basic_data_completion_report.md"
    md.write_text(
        "# v4.3 基础数据补全报告\n\n"
        + "## Summary\n\n"
        + to_md(summary)
        + "\n\n## Coverage\n\n"
        + to_md(report)
        + "\n\n## Notes\n\n"
        + "- 本轮复用既有 v42、Tushare/AkShare、Wind 缓存和本地转债数据库，不做全量重拉。\n"
        + "- 严格信用 AA+/AA 与国开债利差仍未完整取得，第一版使用代理列且标记 `is_proxy=True`。\n"
        + "- 工业利润月频与REITs分派率已接入v43；真实评级下调、分省城投利差仍为非阻塞增强项。\n",
        encoding="utf-8",
    )
    report.to_csv(REPORTS / "v43_basic_data_completion_report.csv", index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    base = build_base_asset_prices()
    rate = build_rate_curves()
    credit = build_credit_spreads(rate)
    funding = build_funding_liquidity()
    cb = build_cb_features()
    equity = build_equity_features()
    macro = build_macro()
    alt = build_alt()

    aligned_daily = align_and_save_daily_outputs(
        {
            "base_asset_prices_daily": base,
            "rate_curves_daily_v43": rate,
            "credit_spreads_daily_v43": credit,
            "funding_liquidity_daily_v43": funding,
            "cb_features_daily_v43": cb,
            "equity_features_daily_v43": equity,
            "alt_features_daily_v43": alt,
        }
    )
    base = aligned_daily["base_asset_prices_daily"]
    rate = aligned_daily["rate_curves_daily_v43"]
    credit = aligned_daily["credit_spreads_daily_v43"]
    funding = aligned_daily["funding_liquidity_daily_v43"]
    cb = aligned_daily["cb_features_daily_v43"]
    equity = aligned_daily["equity_features_daily_v43"]
    alt = aligned_daily["alt_features_daily_v43"]
    macro.to_parquet(PROCESSED / "macro_factors_monthly_v43.parquet", index=False)

    panel_daily, panel_monthly = build_feature_panels()
    aligned_monthly = align_and_save_monthly_outputs(
        {
            "macro_factors_monthly_v43": macro,
            "feature_panel_monthly_v43": panel_monthly,
        }
    )
    macro = aligned_monthly["macro_factors_monthly_v43"]
    panel_monthly = aligned_monthly["feature_panel_monthly_v43"]
    daily_start, daily_end = common_date_window([base], "daily")
    monthly_start = pd.Timestamp(base["date"].min()) + pd.offsets.MonthEnd(0)
    monthly_end = last_completed_month_end(pd.Timestamp(base["date"].max()))
    align_return_tables(daily_start, daily_end, monthly_start, monthly_end)

    datasets = {
        "base_asset_prices_daily": (base, ["date", "asset_code", "close", "return_1d"], True),
        "rate_curves_daily_v43": (rate, ["date", "treasury_y1", "treasury_y3", "treasury_y5", "treasury_y10", "treasury_y30"], True),
        "credit_spreads_daily_v43": (credit, ["date", "spread_aaa_3y_proxy", "spread_aap_3y_proxy", "spread_aa_3y_proxy", "spread_aaa_5y_proxy"], True),
        "funding_liquidity_daily_v43": (funding, ["date", "r007", "dr007", "r007_dr007_spread", "shibor_3m", "aaa_cd_1y_final"], True),
        "cb_features_daily_v43": (cb, ["date", "cb_price_median", "cb_iv_median", "cb_ytm_median", "low_price_ratio_100", "low_price_ratio_110", "low_price_ratio_115", "high_price_premium_ratio", "callable_pressure_ratio"], True),
        "equity_features_daily_v43": (equity, ["date", "csi1000_close", "dividend_close", "sw_l1_positive_5d_ratio", "rzmre"], True),
        "macro_factors_monthly_v43": (macro, ["month", "cpi_yoy", "ppi_yoy", "m2_yoy", "social_financing_new", "industrial_production_yoy", "fixed_asset_investment_yoy", "export_yoy", "effective_date"], True),
        "alt_features_daily_v43": (alt, ["date", "us_10y_yield", "usd_index", "nanhua_commodity", "gold_au9999"], True),
        "feature_panel_daily_v43": (panel_daily, ["date", "rate_treasury_y10", "credit_spread_aaa_3y_proxy", "funding_dr007", "cb_cb_price_median", "equity_sw_l1_positive_5d_ratio"], True),
        "feature_panel_monthly_v43": (panel_monthly, ["month", "rate_treasury_y10", "credit_spread_aaa_3y_proxy", "funding_dr007", "cb_cb_price_median"], True),
        "rating_downgrade_true": (pd.DataFrame(), ["actual_rating_downgrade_count"], False),
        "lgfv_province_spreads": (read_parquet("lgfv_province_spreads_daily_v42.parquet"), ["province", "spread"], False),
    }
    report = write_report(datasets)
    print(report[["dataset", "rows", "start", "end", "quality_flag", "status_bucket", "first_version_blocker", "missing_required"]].to_string(index=False))


if __name__ == "__main__":
    main()
