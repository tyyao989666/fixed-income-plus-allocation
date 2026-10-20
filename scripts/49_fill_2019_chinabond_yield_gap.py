from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_AK = ROOT / "data_raw" / "akshare"
PROCESSED = ROOT / "data_processed"


def fetch_2019_gap() -> pd.DataFrame:
    import akshare as ak

    frames = []
    # AkShare requires the query window to be shorter than one year.
    windows = [("20190101", "20190630"), ("20190701", "20191231")]
    for start, end in windows:
        df = ak.bond_china_yield(start_date=start, end_date=end)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期"]).drop_duplicates(["曲线名称", "日期"]).sort_values(["日期", "曲线名称"])
    RAW_AK.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW_AK / "bond_china_yield_2019_gap.parquet", index=False)
    out.to_csv(RAW_AK / "bond_china_yield_2019_gap.csv", index=False, encoding="utf-8-sig")
    return out


def merge_raw_curves(gap: pd.DataFrame) -> pd.DataFrame:
    raw_path = PROCESSED / "raw_chinabond_yield_curves.parquet"
    raw = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()
    if raw.empty:
        merged = gap
    else:
        raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
        merged = pd.concat([raw, gap], ignore_index=True, sort=False)
        merged = merged.dropna(subset=["日期"]).drop_duplicates(["曲线名称", "日期"], keep="last")
        merged = merged.sort_values(["日期", "曲线名称"])
    merged.to_parquet(PROCESSED / "raw_chinabond_yield_curves.parquet", index=False)
    return merged


def rebuild_long_curves(merged: pd.DataFrame) -> pd.DataFrame:
    cols = {"1年": "y1", "3年": "y3", "5年": "y5", "10年": "y10"}
    curve_map = {
        "中债国债收益率曲线": "treasury",
        "中债中短期票据收益率曲线(AAA)": "mtb_aaa",
        "中债商业银行普通债收益率曲线(AAA)": "bank_bond_aaa",
    }
    rows = []
    for curve_name, key in curve_map.items():
        tmp = merged[merged["曲线名称"].astype(str).eq(curve_name)].copy()
        if tmp.empty:
            continue
        keep = ["日期"] + [c for c in cols if c in tmp.columns]
        tmp = tmp[keep].rename(columns={"日期": "date", **{k: v for k, v in cols.items() if k in tmp.columns}})
        tmp["curve_key"] = key
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out[["date", "curve_key"] + [c for c in ["y1", "y3", "y5", "y10"] if c in out.columns]]
    out = out.sort_values(["date", "curve_key"])
    out.to_parquet(PROCESSED / "credit_yield_curves_daily_v42_long.parquet", index=False)
    return out


def rebuild_spread_tables(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    wide = pd.DataFrame({"date": sorted(long_df["date"].dropna().unique())})
    for key in ["bank_bond_aaa", "mtb_aaa", "treasury"]:
        tmp = long_df[long_df["curve_key"].eq(key)].copy()
        for col in ["y1", "y3", "y5", "y10"]:
            if col in tmp.columns:
                tmp = tmp.rename(columns={col: f"{key}_{col}"})
        wide = wide.merge(tmp.drop(columns=["curve_key"], errors="ignore"), on="date", how="left")
    wide["credit_spread_aaa_1_3y"] = pd.NA
    wide["credit_spread_aa_plus_1_3y"] = pd.NA
    wide["credit_spread_aaa_3_5y"] = pd.NA
    wide["credit_spread_aa_1_3y"] = pd.NA
    wide["source"] = "akshare_chinabond_2019_gap_plus_existing"
    wide.to_parquet(PROCESSED / "credit_spreads_strict_daily_v42.parquet", index=False)
    wide["credit_spread_proxy_aaa_mtb_1_3y_vs_treasury"] = wide["mtb_aaa_y3"] - wide["treasury_y3"]
    wide["credit_spread_proxy_aaa_mtb_3_5y_vs_treasury"] = wide["mtb_aaa_y5"] - wide["treasury_y5"]
    wide["credit_spread_proxy_aaa_bank_1_3y_vs_treasury"] = wide["bank_bond_aaa_y3"] - wide["treasury_y3"]
    wide["credit_spread_proxy_available"] = wide[
        [
            "credit_spread_proxy_aaa_mtb_1_3y_vs_treasury",
            "credit_spread_proxy_aaa_mtb_3_5y_vs_treasury",
            "credit_spread_proxy_aaa_bank_1_3y_vs_treasury",
        ]
    ].notna().any(axis=1)
    wide["strict_missing_reason"] = "缺国开债曲线、AA+中票、AA中票；当前为AAA相对国债代理口径"
    wide.to_parquet(PROCESSED / "credit_spreads_proxy_daily_v42.parquet", index=False)
    wide.to_csv(PROCESSED / "credit_spreads_proxy_daily_v42.csv", index=False, encoding="utf-8-sig")
    return wide


def main() -> None:
    gap = fetch_2019_gap()
    if gap.empty:
        raise RuntimeError("AkShare bond_china_yield returned empty 2019 gap data")
    merged = merge_raw_curves(gap)
    long_df = rebuild_long_curves(merged)
    spreads = rebuild_spread_tables(long_df)
    print(
        {
            "gap_rows": len(gap),
            "merged_rows": len(merged),
            "long_rows": len(long_df),
            "spread_rows": len(spreads),
            "spread_start": str(spreads["date"].min().date()),
            "spread_end": str(spreads["date"].max().date()),
        }
    )


if __name__ == "__main__":
    main()
