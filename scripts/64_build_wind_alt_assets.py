from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
RAW_REITS = ROOT / "data_raw" / "wind" / "reits_equal_weight"

NANHUA_ASSET_MAP: dict[str, dict[str, str]] = {
    "nanhua_commodity": {"asset_code": "alt_commodity", "asset_name": "南华商品"},
    "nanhua_nonferrous": {"asset_code": "alt_nonferrous", "asset_name": "南华有色"},
    "nanhua_chemical": {"asset_code": "alt_chemical", "asset_name": "南华能化"},
}

REITS_POOL: dict[str, str] = {
    "180101.SZ": "博时蛇口产园REIT",
    "180201.SZ": "平安广州广河REIT",
    "180301.SZ": "红土盐田港REIT",
    "180801.SZ": "中航首钢绿能REIT",
    "508000.SH": "华安张江光大REIT",
    "508001.SH": "浙商沪杭甬REIT",
    "508006.SH": "富国首创水务REIT",
    "508027.SH": "东吴苏园产业REIT",
    "508056.SH": "中金普洛斯REIT",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _asset_template(asset_code: str, asset_name: str, source: str, proxy: bool = False) -> dict[str, Any]:
    return {
        "asset_code": asset_code,
        "code": asset_code,
        "asset_name": asset_name,
        "asset_class": "alternatives",
        "module": "alternatives",
        "sub_module": asset_code,
        "data_source": source,
        "fetch_time": _now(),
        "is_proxy": proxy,
    }


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    return [x.strip() for x in raw.split(";") if x.strip()] + [None]


def parse_kline_payload(payload: dict[str, Any] | list[Any]) -> pd.DataFrame:
    if isinstance(payload, dict) and "columns" in payload and "rows" in payload:
        cols = [str(c.get("name", f"col_{i}")) for i, c in enumerate(payload.get("columns") or [])]
        rows = payload.get("rows") or []
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols[: len(rows[0])])
    return pd.DataFrame(payload)


def call_wind_fund_kline(wind_code: str, start: str, end: str, cache: Path) -> tuple[pd.DataFrame, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8")).get("data", {})
        return parse_kline_payload(data), "ok_cached"

    params = {"windcode": wind_code, "begin_date": start, "end_date": end, "period": "10"}
    last_error: str | None = None
    for key in key_pool():
        env = os.environ.copy()
        if key:
            env["WIND_API_KEY"] = key
        res = subprocess.run(
            ["node", str(SKILL_CLI), "call", "fund_data", "get_fund_kline", json.dumps(params, ensure_ascii=False)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
            env=env,
        )
        if res.returncode != 0:
            last_error = res.stderr[-500:] or res.stdout[-500:]
            continue
        try:
            outer = json.loads(res.stdout)
            payload = json.loads(outer["content"][0]["text"])
        except Exception as exc:
            last_error = f"{exc}: {res.stdout[:500]}"
            continue
        if payload.get("error"):
            last_error = json.dumps(payload.get("error"), ensure_ascii=False)
            if "BALANCE_INSUFFICIENT" in last_error:
                continue
            return pd.DataFrame(), f"failed:{last_error}"
        data = payload.get("data") or {}
        cache.write_text(json.dumps({"params": params, "data": data}, ensure_ascii=False, indent=2), encoding="utf-8")
        return parse_kline_payload(data), "ok"
    return pd.DataFrame(), f"failed:{last_error or 'unknown_error'}"


def default_monthly_end() -> pd.Timestamp:
    cfg_path = ROOT / "config" / "v1_strategy_config.yaml"
    if cfg_path.exists():
        text = cfg_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("end_date:"):
                return pd.Timestamp(line.split(":", 1)[1].strip().strip('"').strip("'"))
    return pd.Timestamp("2026-04-30")


def build_nanhua_assets(monthly_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = PROCESSED / "wind_nanhua_alt_indices_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/63_fetch_wind_nanhua_alt_indices.py first.")
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    daily_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []

    for alias, asset in NANHUA_ASSET_MAP.items():
        sub = raw[raw["alias"].astype(str).eq(alias)].copy()
        if sub.empty:
            continue
        meta = _asset_template(asset["asset_code"], asset["asset_name"], f"wind_nanhua:{alias}")
        daily = pd.DataFrame(
            {
                "date": sub["date"],
                **meta,
                "ts_code": sub["wind_code"],
                "open": pd.to_numeric(sub["open"], errors="coerce"),
                "high": pd.to_numeric(sub["high"], errors="coerce"),
                "low": pd.to_numeric(sub["low"], errors="coerce"),
                "close": pd.to_numeric(sub["close"], errors="coerce"),
                "vol": pd.to_numeric(sub["volume"], errors="coerce"),
                "amount": pd.to_numeric(sub["turnover"], errors="coerce"),
            }
        )
        daily = daily.dropna(subset=["date", "close"]).sort_values("date")
        daily["return_1d"] = daily["close"].pct_change()
        daily_frames.append(daily)

        month_last = daily.set_index("date").resample("ME").last().dropna(subset=["close"]).reset_index()
        month_last = month_last[month_last["date"].le(monthly_end)].copy()
        month_last["return_1m"] = month_last["close"].pct_change()
        monthly = month_last[
            [
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
        ].copy()
        monthly_frames.append(monthly)

    daily_all = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    monthly_all = pd.concat(monthly_frames, ignore_index=True, sort=False) if monthly_frames else pd.DataFrame()
    return daily_all, monthly_all


def build_us_bond_proxy(duration: float, cap: float, monthly_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = PROCESSED / "alt_overseas_filled_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    y = pd.to_numeric(raw["us_10y_yield"], errors="coerce")
    df = pd.DataFrame({"date": raw["date"], "yield_pct": y}).dropna(subset=["date", "yield_pct"]).sort_values("date")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    dy = df["yield_pct"].diff() / 100.0
    carry = df["yield_pct"].shift(1) / 100.0 / 252.0
    ret = (carry - duration * dy).replace([np.inf, -np.inf], np.nan)
    ret = ret.where(ret.abs().le(cap))
    close = (1.0 + ret.fillna(0.0)).cumprod() * 100.0
    meta = _asset_template("alt_us_bond", "美债10Y曲线代理", f"wind_us_10y_curve_proxy:duration_{duration:g}", proxy=True)
    daily = pd.DataFrame(
        {
            "date": df["date"],
            **meta,
            "ts_code": "USGG10YR",
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": close,
            "vol": np.nan,
            "amount": np.nan,
            "return_1d": ret,
            "yield_pct": df["yield_pct"],
        }
    )

    month_y = df.set_index("date")["yield_pct"].resample("ME").last().dropna()
    m_dy = month_y.diff() / 100.0
    m_carry = month_y.shift(1) / 100.0 / 12.0
    m_ret = (m_carry - duration * m_dy).replace([np.inf, -np.inf], np.nan)
    m_ret = m_ret.where(m_ret.abs().le(cap * 5))
    m_close = (1.0 + m_ret.fillna(0.0)).cumprod() * 100.0
    monthly = pd.DataFrame(
        {
            "date": month_y.index,
            **meta,
            "close": m_close.to_numpy(dtype=float),
            "return_1m": m_ret.to_numpy(dtype=float),
        }
    )
    monthly = monthly[monthly["date"].le(monthly_end)].copy()
    return daily, monthly


def build_reits_equal_weight(monthly_end: pd.Timestamp, start: str = "20210601") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end = monthly_end.strftime("%Y%m%d")
    member_frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []
    for code, name in REITS_POOL.items():
        cache = RAW_REITS / f"{code}_{start}_{end}.json"
        raw, status = call_wind_fund_kline(code, start, end, cache)
        if raw.empty:
            report_rows.append({"wind_code": code, "name": name, "status": status, "rows": 0, "start": "", "end": ""})
            continue
        date_col = "_DATE" if "_DATE" in raw.columns else "TIME"
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col], errors="coerce"),
                "wind_code": code,
                "member_name": name,
                "close": pd.to_numeric(raw.get("MATCH"), errors="coerce"),
                "volume": pd.to_numeric(raw.get("VOLUME"), errors="coerce"),
                "turnover": pd.to_numeric(raw.get("TURNOVER"), errors="coerce"),
            }
        )
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        trading = df["volume"].fillna(0).gt(0) | df["turnover"].fillna(0).gt(0)
        if trading.any():
            df = df.loc[trading].copy()
        df["member_return_1d"] = df.groupby("wind_code")["close"].pct_change()
        member_frames.append(df)
        report_rows.append(
            {
                "wind_code": code,
                "name": name,
                "status": status,
                "rows": int(len(df)),
                "start": df["date"].min().date().isoformat() if not df.empty else "",
                "end": df["date"].max().date().isoformat() if not df.empty else "",
            }
        )

    member = pd.concat(member_frames, ignore_index=True, sort=False) if member_frames else pd.DataFrame()
    report = pd.DataFrame(report_rows)
    if member.empty:
        return pd.DataFrame(), pd.DataFrame(), report

    ret = (
        member.dropna(subset=["member_return_1d"])
        .groupby("date")
        .agg(return_1d=("member_return_1d", "mean"), member_count=("wind_code", "nunique"), close_raw=("close", "mean"))
        .reset_index()
        .sort_values("date")
    )
    ret = ret[ret["member_count"].ge(5)].copy()
    ret["close"] = (1.0 + ret["return_1d"].fillna(0.0)).cumprod() * 100.0
    meta = _asset_template("alt_reits", "公募REITs等权", "wind_reits_equal_weight:first_9_reits", proxy=True)
    daily = pd.DataFrame(
        {
            "date": ret["date"],
            **meta,
            "ts_code": "REITS_EQUAL_WEIGHT",
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": ret["close"],
            "vol": np.nan,
            "amount": np.nan,
            "return_1d": ret["return_1d"],
            "member_count": ret["member_count"],
        }
    )

    month_last = daily.set_index("date").resample("ME").last().dropna(subset=["close"]).reset_index()
    month_last = month_last[month_last["date"].le(monthly_end)].copy()
    month_last["return_1m"] = month_last["close"].pct_change()
    monthly = month_last[
        [
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
    ].copy()
    return daily, monthly, report


def merge_asset_returns(new_daily: pd.DataFrame, new_monthly: pd.DataFrame, apply: bool) -> pd.DataFrame:
    daily_path = PROCESSED / "asset_returns_daily.parquet"
    monthly_path = PROCESSED / "asset_returns_monthly.parquet"
    daily = pd.read_parquet(daily_path)
    monthly = pd.read_parquet(monthly_path)
    replace_codes = sorted(set(new_daily["asset_code"].dropna().astype(str)) | set(new_monthly["asset_code"].dropna().astype(str)))

    old_summary = summarize_assets(monthly[monthly["asset_code"].astype(str).isin(replace_codes)], "old")
    new_summary = summarize_assets(new_monthly, "new")
    report = pd.concat([old_summary, new_summary], ignore_index=True, sort=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS / "wind_alt_asset_replacement_report.csv", index=False, encoding="utf-8-sig")
    new_summary.to_csv(REPORTS / "wind_alt_asset_final_coverage.csv", index=False, encoding="utf-8-sig")

    if apply:
        daily_out = pd.concat(
            [daily[~daily["asset_code"].astype(str).isin(replace_codes)], new_daily[daily.columns.intersection(new_daily.columns)]],
            ignore_index=True,
            sort=False,
        )
        for c in daily.columns:
            if c not in daily_out.columns:
                daily_out[c] = np.nan
        daily_out = daily_out[daily.columns].sort_values(["date", "asset_code"]).reset_index(drop=True)

        monthly_out = pd.concat(
            [monthly[~monthly["asset_code"].astype(str).isin(replace_codes)], new_monthly[monthly.columns.intersection(new_monthly.columns)]],
            ignore_index=True,
            sort=False,
        )
        for c in monthly.columns:
            if c not in monthly_out.columns:
                monthly_out[c] = np.nan
        monthly_out = monthly_out[monthly.columns].sort_values(["date", "asset_code"]).reset_index(drop=True)
        daily_out.to_parquet(daily_path, index=False)
        monthly_out.to_parquet(monthly_path, index=False)
    return report


def summarize_assets(df: pd.DataFrame, version: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = []
    for code, g in df.copy().groupby("asset_code"):
        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        r = pd.to_numeric(g.get("return_1m"), errors="coerce")
        out.append(
            {
                "version": version,
                "asset_code": code,
                "asset_name": g["asset_name"].dropna().astype(str).iloc[-1] if g["asset_name"].notna().any() else "",
                "rows": int(len(g)),
                "start": g["date"].min().date().isoformat(),
                "end": g["date"].max().date().isoformat(),
                "ann_return": float((1 + r.fillna(0)).prod() ** (12 / max(len(r), 1)) - 1),
                "ann_vol": float(r.std(skipna=True) * np.sqrt(12)),
                "max_abs_monthly_return": float(r.abs().max(skipna=True)),
                "data_source": g["data_source"].dropna().astype(str).iloc[-1] if g["data_source"].notna().any() else "",
            }
        )
    return pd.DataFrame(out)


def update_alt_feature_panels(nanhua_daily: pd.DataFrame, us_bond_daily: pd.DataFrame, apply: bool) -> None:
    path = PROCESSED / "alt_features_daily_v43.parquet"
    if not apply or not path.exists():
        return
    features = pd.read_parquet(path)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")

    wide = pd.DataFrame()
    if not nanhua_daily.empty:
        wide = nanhua_daily.pivot_table(index="date", columns="asset_code", values="close", aggfunc="last").reset_index()
        wide = wide.rename(
            columns={
                "alt_commodity": "nanhua_commodity",
                "alt_nonferrous": "nanhua_nonferrous",
                "alt_chemical": "nanhua_chemical",
            }
        )
    if not us_bond_daily.empty:
        ub = us_bond_daily[["date", "yield_pct", "return_1d"]].rename(
            columns={"yield_pct": "us_10y_yield", "return_1d": "us_bond_curve_proxy_ret_1d"}
        )
        wide = ub if wide.empty else wide.merge(ub, on="date", how="outer")
    if wide.empty:
        return

    merged = features.drop(columns=[c for c in wide.columns if c != "date" and c in features.columns], errors="ignore").merge(
        wide, on="date", how="outer"
    )
    merged = merged.sort_values("date")
    for c in ["nanhua_commodity", "nanhua_nonferrous", "nanhua_chemical"]:
        if c in merged.columns:
            merged[f"{c}_ret_1d"] = pd.to_numeric(merged[c], errors="coerce").pct_change(fill_method=None)
    merged.to_parquet(path, index=False)


def update_asset_universe_parquet(apply: bool) -> None:
    if not apply:
        return
    cfg_path = ROOT / "config" / "asset_universe.yaml"
    if not cfg_path.exists():
        return
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    assets = pd.DataFrame(cfg.get("assets", []))
    if assets.empty:
        return
    if "asset_code" not in assets.columns and "code" in assets.columns:
        assets.insert(0, "asset_code", assets["code"].astype(str))
    assets.to_parquet(PROCESSED / "asset_universe.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean alternative assets from Wind data.")
    parser.add_argument("--duration", type=float, default=7.5, help="Modified duration for US 10Y bond curve proxy.")
    parser.add_argument("--daily-return-cap", type=float, default=0.20)
    parser.add_argument("--monthly-end", default=None, help="Monthly asset cutoff, default v1 rebalance end_date.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    monthly_end = pd.Timestamp(args.monthly_end) if args.monthly_end else default_monthly_end()
    nanhua_daily, nanhua_monthly = build_nanhua_assets(monthly_end)
    us_daily, us_monthly = build_us_bond_proxy(args.duration, args.daily_return_cap, monthly_end)
    reits_daily, reits_monthly, reits_report = build_reits_equal_weight(monthly_end)
    reits_report.to_csv(REPORTS / "wind_reits_equal_weight_fetch_report.csv", index=False, encoding="utf-8-sig")
    new_daily = pd.concat([nanhua_daily, us_daily, reits_daily], ignore_index=True, sort=False)
    new_monthly = pd.concat([nanhua_monthly, us_monthly, reits_monthly], ignore_index=True, sort=False)
    report = merge_asset_returns(new_daily, new_monthly, apply=not args.dry_run)
    update_alt_feature_panels(nanhua_daily, us_daily, apply=not args.dry_run)
    update_asset_universe_parquet(apply=not args.dry_run)
    print(
        {
            "apply": not args.dry_run,
            "new_daily_rows": int(len(new_daily)),
            "new_monthly_rows": int(len(new_monthly)),
            "assets": sorted(new_monthly["asset_code"].dropna().astype(str).unique().tolist()),
            "monthly_end": monthly_end.strftime("%Y-%m-%d"),
            "report": str((REPORTS / "wind_alt_asset_replacement_report.csv").relative_to(ROOT)),
            "report_rows": int(len(report)),
        }
    )


if __name__ == "__main__":
    main()
