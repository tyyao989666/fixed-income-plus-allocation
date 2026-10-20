from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
RAW = ROOT / "data_raw"
REPORTS = ROOT / "output" / "reports"


def _base_meta(existing: pd.DataFrame, asset_code: str) -> dict[str, object]:
    row = existing[existing["asset_code"].astype(str).eq(asset_code)].iloc[-1].to_dict()
    return {key: row.get(key) for key in existing.columns if key not in {"date", "close", "return_1d", "return_1m"}}


def _fit_schema(frame: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[list(columns)]


def build_us_bond(existing_daily: pd.DataFrame, fred_csv: Path, start: pd.Timestamp, end: pd.Timestamp, duration: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    wind = pd.read_parquet(PROCESSED / "raw_wind_us_10y_yield_daily.parquet")
    wind_yield = pd.Series(
        pd.to_numeric(wind["美国:国债收益率:10年"], errors="coerce").to_numpy(),
        index=pd.to_datetime(wind["日期"], errors="coerce"),
        name="wind_yield",
    ).dropna().sort_index()
    fred = pd.read_csv(fred_csv)
    fred_date = pd.to_datetime(fred.iloc[:, 0], errors="coerce")
    fred_yield = pd.Series(pd.to_numeric(fred.iloc[:, 1], errors="coerce").to_numpy(), index=fred_date, name="fred_yield").dropna().sort_index()
    joined = pd.concat([wind_yield, fred_yield], axis=1).sort_index()
    joined["yield_pct"] = joined["wind_yield"].combine_first(joined["fred_yield"])
    joined["source_component"] = np.where(joined["wind_yield"].notna(), "wind", "fred_gap_fill")
    joined = joined.loc[(joined.index >= start - pd.DateOffset(months=1)) & (joined.index <= end)].dropna(subset=["yield_pct"])

    daily_ret = joined["yield_pct"].shift(1) / 100.0 / 252.0 - duration * joined["yield_pct"].diff() / 100.0
    daily_close = (1.0 + daily_ret.fillna(0.0)).cumprod() * 100.0
    meta = _base_meta(existing_daily, "alt_us_bond")
    meta.update(
        {
            "asset_code": "alt_us_bond",
            "code": "alt_us_bond",
            "data_source": f"wind_DGS10+FRED_DGS10_gap_fill:duration_{duration:g}",
            "is_proxy": True,
        }
    )
    daily = pd.DataFrame(
        {
            "date": joined.index,
            **meta,
            "close": daily_close.to_numpy(),
            "return_1d": daily_ret.to_numpy(),
            "yield_pct": joined["yield_pct"].to_numpy(),
            "source_component": joined["source_component"].to_numpy(),
        }
    )
    daily = daily[daily["date"].between(start, end)].copy()

    month_yield = joined["yield_pct"].resample("ME").last().dropna()
    monthly_ret = month_yield.shift(1) / 100.0 / 12.0 - duration * month_yield.diff() / 100.0
    monthly_close = (1.0 + monthly_ret.fillna(0.0)).cumprod() * 100.0
    monthly = pd.DataFrame(
        {
            "date": month_yield.index,
            **meta,
            "close": monthly_close.to_numpy(),
            "return_1m": monthly_ret.to_numpy(),
        }
    )
    monthly = monthly[monthly["date"].between(start, end)].copy()
    return daily, monthly


def build_csi1000(existing_daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(RAW / "tushare" / "index_daily_csi1000_20180101_20260603.parquet")
    raw["date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").copy()
    raw["return_1d"] = pd.to_numeric(raw["pct_chg"], errors="coerce") / 100.0
    meta = _base_meta(existing_daily, "equity_csi1000")
    meta.update(
        {
            "asset_code": "equity_csi1000",
            "code": "equity_csi1000",
            "asset_name": "中证1000全收益价格代理",
            "data_source": "tushare:000852.SH:index_close",
            "is_proxy": True,
        }
    )
    daily = pd.DataFrame(
        {
            "date": raw["date"],
            **meta,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "vol": pd.to_numeric(raw["vol"], errors="coerce"),
            "amount": pd.to_numeric(raw["amount"], errors="coerce"),
            "return_1d": raw["return_1d"],
        }
    )
    daily = daily[daily["date"].between(start, end)].copy()

    monthly = daily.set_index("date").resample("ME").last().dropna(subset=["close"]).reset_index()
    first_pre_close = float(raw.loc[raw["date"].ge(start), "pre_close"].iloc[0])
    prior = monthly["close"].shift(1)
    prior.iloc[0] = first_pre_close
    monthly["return_1m"] = monthly["close"] / prior - 1.0
    return daily, monthly


def summarize(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for code, group in frame.groupby("asset_code"):
        ret_col = "return_1m" if "return_1m" in group.columns else "return_1d"
        values = pd.to_numeric(group[ret_col], errors="coerce")
        rows.append(
            {
                "version": label,
                "asset_code": code,
                "start": pd.to_datetime(group["date"]).min(),
                "end": pd.to_datetime(group["date"]).max(),
                "rows": len(group),
                "missing_returns": int(values.isna().sum()),
                "max_abs_return": float(values.abs().max()),
                "data_source": str(group.get("data_source", pd.Series([""])).iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair investable return series without look-ahead interpolation.")
    parser.add_argument("--fred-csv", default="data_raw/fred/DGS10_2017-12-01_2026-04-30.csv")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--duration", type=float, default=7.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    fred_csv = ROOT / args.fred_csv
    if not fred_csv.exists():
        raise FileNotFoundError(f"Download FRED DGS10 first: {fred_csv}")
    daily_path = PROCESSED / "asset_returns_daily.parquet"
    monthly_path = PROCESSED / "asset_returns_monthly.parquet"
    old_daily = pd.read_parquet(daily_path)
    old_monthly = pd.read_parquet(monthly_path)
    codes = ["alt_us_bond", "equity_csi1000"]

    us_daily, us_monthly = build_us_bond(old_daily, fred_csv, start, end, args.duration)
    csi_daily, csi_monthly = build_csi1000(old_daily, start, end)
    new_daily = pd.concat([us_daily, csi_daily], ignore_index=True, sort=False)
    new_monthly = pd.concat([us_monthly, csi_monthly], ignore_index=True, sort=False)
    report = pd.concat(
        [
            summarize(old_monthly[old_monthly["asset_code"].isin(codes)], "before"),
            summarize(new_monthly, "after"),
        ],
        ignore_index=True,
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS / "v1_return_series_repair_report.csv", index=False, encoding="utf-8-sig")

    if not args.dry_run:
        daily_out = pd.concat(
            [old_daily[~old_daily["asset_code"].isin(codes)], _fit_schema(new_daily, old_daily.columns)],
            ignore_index=True,
        ).sort_values(["date", "asset_code"]).reset_index(drop=True)
        monthly_out = pd.concat(
            [old_monthly[~old_monthly["asset_code"].isin(codes)], _fit_schema(new_monthly, old_monthly.columns)],
            ignore_index=True,
        ).sort_values(["date", "asset_code"]).reset_index(drop=True)
        daily_out.to_parquet(daily_path, index=False)
        monthly_out.to_parquet(monthly_path, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
