from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
RAW = ROOT / "data_raw"
REPORTS = ROOT / "output" / "reports"
IFIND_CALL_CANDIDATES = [
    Path.home() / ".agents" / "skills" / "ifind-finance-data" / "call-node.js",
    Path.home() / ".codex" / "skills" / "ifinD-finance-data" / "call-node.js",
]
IFIND_CALL = next((path for path in IFIND_CALL_CANDIDATES if path.exists()), IFIND_CALL_CANDIDATES[0])

REITS = [
    "180101.SZ",
    "180201.SZ",
    "180301.SZ",
    "180801.SZ",
    "508000.SH",
    "508001.SH",
    "508006.SH",
    "508027.SH",
    "508056.SH",
]

CREDIT_INDEX_ASSETS = {
    "931740.CSI": "credit_lgfv_aa_plus",
    "H11098.CSI": "credit_lgfv_aa",
    "CBA04201.CB": "credit_ind_aaa",
    "CBA04101.CB": "credit_ind_aa_plus",
    "CBA04001.CB": "credit_ind_aa",
    "CBA00621.CB": "rate_1_3y",
    "CBA00631.CB": "rate_3_5y",
    "CBA00651.CB": "rate_7_10y",
    "CBA00661.CB": "rate_30y",
}

CB_STYLE_NAMES = {
    "bond": "债性",
    "balanced": "平衡",
    "equity": "股性",
    "large": "大盘",
    "small": "小盘",
    "low": "低价",
    "high": "高价",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def read_parquet(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_parquet(df: pd.DataFrame, name: str, dry_run: bool) -> None:
    if dry_run:
        return
    path = PROCESSED / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def backup_files(names: list[str], stamp: str, dry_run: bool) -> Path:
    target = RAW / "refresh_backups" / stamp
    if dry_run:
        return target
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = PROCESSED / name
        dst = target / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return target


def clean_date(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    out = pd.to_datetime(text, errors="coerce")
    if compact.any():
        out.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return out


def completed_month_end(latest_trade_date: pd.Timestamp) -> pd.Timestamp:
    today = pd.Timestamp.today().normalize()
    if latest_trade_date.to_period("M") < today.to_period("M"):
        return latest_trade_date.to_period("M").to_timestamp("M")
    return (today.to_period("M") - 1).to_timestamp("M")


def tushare_call(pro: Any, endpoint: str, retries: int = 3, **params: Any) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            result = getattr(pro, endpoint)(**params)
            return result if result is not None else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - depends on the provider
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Tushare {endpoint} failed after {retries} attempts: {last}")


def latest_complete_trade_date(pro: Any, as_of: pd.Timestamp) -> pd.Timestamp:
    start = (as_of - pd.Timedelta(days=15)).strftime("%Y%m%d")
    probes = [
        ("fund_daily", "510300.SH"),
        ("fund_daily", "510880.SH"),
        ("index_daily", "000300.SH"),
    ]
    observed: list[pd.Timestamp] = []
    for endpoint, code in probes:
        sample = tushare_call(
            pro,
            endpoint,
            ts_code=code,
            start_date=start,
            end_date=as_of.strftime("%Y%m%d"),
        )
        if not sample.empty and "trade_date" in sample:
            last = clean_date(sample["trade_date"]).max()
            if pd.notna(last):
                observed.append(last.normalize())
        if observed:
            break
        time.sleep(1.0)
    if observed:
        return max(observed)

    # A transient provider empty response must not turn the wall-clock date into
    # a fictitious close.  Reuse only a previously persisted raw market bar.
    raw_dir = RAW / "tushare" / "latest_refresh"
    for path in raw_dir.glob("*.parquet"):
        try:
            cached = pd.read_parquet(path, columns=["trade_date"])
        except Exception:
            continue
        if not cached.empty:
            last = clean_date(cached["trade_date"]).max()
            if pd.notna(last) and last.normalize() <= as_of:
                observed.append(last.normalize())
    if not observed:
        raise RuntimeError("Cannot determine the latest completed market close from Tushare or persisted raw bars.")
    return max(observed)


def _markdown_table(text: str) -> pd.DataFrame:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return pd.DataFrame()
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    header = rows[0]
    data = [row for row in rows[2:] if len(row) == len(header)]
    return pd.DataFrame(data, columns=header)


def _unwrap_json_object(value: Any) -> Any:
    """Unwrap one or more JSON-encoded response layers from iFinD gateways."""
    while isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def parse_ifind_response(response: dict[str, Any] | str) -> pd.DataFrame:
    response = _unwrap_json_object(response)
    if not isinstance(response, dict):
        raise RuntimeError(f"iFinD response is not a JSON object: {str(response)[:240]}")
    if not response.get("ok"):
        raise RuntimeError(f"iFinD call failed: {response.get('error')}")
    outer = _unwrap_json_object(response.get("data", {}))
    if not isinstance(outer, dict):
        raise RuntimeError(f"iFinD data wrapper is not an object: {str(outer)[:240]}")
    result = _unwrap_json_object(outer.get("result", {}))
    if not isinstance(result, dict):
        raise RuntimeError(f"iFinD result wrapper is not an object: {str(result)[:240]}")
    content = _unwrap_json_object(result.get("content", []))
    if isinstance(content, dict):
        content = [content]
    if not content:
        raise RuntimeError("iFinD returned no content.")
    first = content[0]
    text = first.get("text", "{}") if isinstance(first, dict) else first
    payload = _unwrap_json_object(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"iFinD content is not a JSON object: {str(payload)[:240]}")
    if payload.get("code") != 1:
        raise RuntimeError(f"iFinD payload error: {payload.get('msg')} {payload.get('subMsg')}")
    body = _unwrap_json_object(payload.get("data") or {})
    if not isinstance(body, dict):
        raise RuntimeError(f"iFinD payload data is not an object: {str(body)[:240]}")
    for block in body.get("datas") or []:
        data = block.get("data") or {}
        rows = data.get("data") or []
        columns = data.get("columns") or []
        if rows and columns:
            return pd.DataFrame(rows, columns=columns)
    table = body.get("text") or body.get("answer") or ""
    frame = _markdown_table(table)
    if frame.empty or "查询结果为空" in table or "工具调用结果为空" in table:
        raise RuntimeError(f"iFinD returned an empty table: {table[:160]}")
    return frame


def ifind_query(
    server: str,
    tool: str,
    query: str,
    cache_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    if not IFIND_CALL.exists():
        raise FileNotFoundError(f"Missing iFinD connector: {IFIND_CALL}")
    digest = hashlib.sha1(f"{server}|{tool}|{query}".encode("utf-8")).hexdigest()[:12]
    cache = cache_dir / f"{server}_{tool}_{digest}.json"
    if cache.exists():
        response = json.loads(cache.read_text(encoding="utf-8"))
        try:
            return parse_ifind_response(response)
        except RuntimeError:
            cache.unlink(missing_ok=True)
    js = (
        "const {call}=require(process.argv[1]);"
        "call(process.argv[2],process.argv[3],JSON.parse(process.argv[4]))"
        ".then(r=>console.log(JSON.stringify(r)))"
        ".catch(e=>{console.error(e.stack||e.message);process.exit(1)})"
    )
    cmd = [
        "node",
        "-e",
        js,
        str(IFIND_CALL),
        server,
        tool,
        json.dumps({"query": query}, ensure_ascii=False),
    ]
    last_error: Exception | None = None
    for attempt in range(3):
        result = subprocess.run(
            cmd,
            cwd=IFIND_CALL.parent,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            last_error = RuntimeError(f"iFinD subprocess failed: {(result.stderr or result.stdout)[-800:]}")
        else:
            try:
                response = json.loads(result.stdout)
                parsed = parse_ifind_response(response)
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
                return parsed
            except (json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"iFinD query failed after 3 attempts: {last_error}")


def date_column(df: pd.DataFrame) -> str:
    match = next((str(c) for c in df.columns if "日期" in str(c) or str(c).lower() in {"date", "month"}), None)
    if match is None:
        raise ValueError(f"No date column in {list(df.columns)}")
    return match


def numeric_column(df: pd.DataFrame, include: list[str], exclude: list[str] | None = None) -> str:
    exclude = exclude or []
    for column in df.columns:
        name = str(column)
        if all(word in name for word in include) and not any(word in name for word in exclude):
            return name
    raise KeyError(f"No column matching include={include}, exclude={exclude}: {list(df.columns)}")


def series_frame(df: pd.DataFrame, value_column: str, output_name: str) -> pd.DataFrame:
    dcol = date_column(df)
    out = pd.DataFrame(
        {
            "date": clean_date(df[dcol]),
            output_name: pd.to_numeric(df[value_column], errors="coerce"),
        }
    )
    return out.dropna(subset=["date", output_name]).drop_duplicates("date", keep="last").sort_values("date")


def numeric_series(df: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.Series:
    """Return an index-aligned numeric series even when a provider omits a field."""
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if primary in df.columns:
        result = pd.to_numeric(df[primary], errors="coerce")
    if fallback and fallback in df.columns:
        result = result.combine_first(pd.to_numeric(df[fallback], errors="coerce"))
    return result


def curve_frame(df: pd.DataFrame, curve_key: str) -> pd.DataFrame:
    dcol = date_column(df)
    out = pd.DataFrame({"date": clean_date(df[dcol])})
    for tenor in [1, 3, 5, 10, 30]:
        pattern = re.compile(rf"(?:^|:)\s*{tenor}年(?:$|（)")
        column = next((str(c) for c in df.columns if pattern.search(str(c))), None)
        if column:
            out[f"y{tenor}"] = pd.to_numeric(df[column], errors="coerce")
    out["curve_key"] = curve_key
    return out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")


def upsert(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()
    columns = list(dict.fromkeys(list(existing.columns) + list(new.columns)))
    old = existing.reindex(columns=columns)
    fresh = new.reindex(columns=columns)
    out = pd.concat([old, fresh], ignore_index=True, sort=False)
    return out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def asset_meta(existing: pd.DataFrame, code: str) -> dict[str, Any]:
    sub = existing[existing["asset_code"].astype(str).eq(code)]
    if sub.empty:
        return {"asset_code": code, "code": code, "asset_name": code}
    row = sub.iloc[-1]
    return {column: row.get(column) for column in existing.columns if column not in {"date", "open", "high", "low", "close", "vol", "amount", "return_1d", "member_count"}}


def fit_asset_schema(frame: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[list(columns)]


def normalize_tushare_asset(raw: pd.DataFrame, meta: dict[str, Any], endpoint: str, fetch_time: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": clean_date(raw["trade_date"])})
    for column in ["open", "high", "low", "close", "vol", "amount"]:
        out[column] = pd.to_numeric(raw.get(column), errors="coerce")
    for key, value in meta.items():
        out[key] = value
    out["fetch_time"] = fetch_time
    out["data_source"] = f"tushare:{endpoint}:incremental"
    return out.dropna(subset=["date", "close"]).sort_values("date")


def refresh_tushare_assets(
    pro: Any,
    existing: pd.DataFrame,
    config: dict[str, Any],
    latest: pd.Timestamp,
    raw_dir: Path,
    status: list[dict[str, Any]],
) -> pd.DataFrame:
    fetch_time = datetime.now().isoformat(timespec="seconds")
    result = existing.copy()
    for asset in config.get("assets", []):
        if asset.get("source_type") != "fund_daily":
            continue
        code = str(asset["code"])
        sub = result[result["asset_code"].astype(str).eq(code)].copy()
        old_end = pd.to_datetime(sub["date"], errors="coerce").max() if not sub.empty else pd.Timestamp("2018-01-01")
        start = max(pd.Timestamp("2018-01-01"), old_end - pd.Timedelta(days=14))
        endpoint = "index_daily" if code == "equity_csi1000" else "fund_daily"
        ts_code = "000852.SH" if code == "equity_csi1000" else str(asset["ts_code"])
        raw = tushare_call(
            pro,
            endpoint,
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=latest.strftime("%Y%m%d"),
        )
        raw_path = raw_dir / f"{endpoint}_{ts_code.replace('.', '_')}_{start:%Y%m%d}_{latest:%Y%m%d}.parquet"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(raw_path, index=False)
        meta = asset_meta(result, code)
        meta.update(
            {
                "asset_code": code,
                "code": code,
                "asset_name": asset.get("name", meta.get("asset_name", code)),
                "asset_class": asset.get("module"),
                "module": asset.get("module"),
                "sub_module": code,
                "ts_code": ts_code,
            }
        )
        fresh = normalize_tushare_asset(raw, meta, endpoint, fetch_time)
        merged = upsert(sub, fresh, ["asset_code", "date"])
        result = pd.concat([result[~result["asset_code"].astype(str).eq(code)], fit_asset_schema(merged, result.columns)], ignore_index=True)
        refresh_status = "ok"
        if fresh.empty:
            refresh_status = "preserved_current" if pd.notna(old_end) and old_end.normalize() >= latest.normalize() else "empty"
        status.append(
            {
                "dataset": f"asset:{code}",
                "source": f"tushare:{endpoint}",
                "end": fresh["date"].max() if not fresh.empty else old_end,
                "rows_added_or_refreshed": len(fresh),
                "status": refresh_status,
            }
        )
        time.sleep(0.25)
    return result


def refresh_credit_index_assets(
    existing: pd.DataFrame,
    cache_dir: Path,
    timeout: int,
    latest: pd.Timestamp,
    status: list[dict[str, Any]],
) -> pd.DataFrame:
    start = max(pd.Timestamp("2026-05-15"), latest - pd.Timedelta(days=45))
    date_range = f"{start:%Y-%m-%d}至{latest:%Y-%m-%d}"
    queries = [
        f"931740.CSI、H11098.CSI在{date_range}的每日收盘点位和涨跌幅",
        f"CBA04201.CB、CBA04101.CB、CBA04001.CB在{date_range}的每日收盘点位和涨跌幅",
        f"CBA00621.CB、CBA00631.CB、CBA00651.CB、CBA00661.CB在{date_range}的每日收盘点位和涨跌幅",
    ]
    frames = [ifind_query("index", "index_data", query, cache_dir, timeout) for query in queries]
    raw = pd.concat(frames, ignore_index=True, sort=False)
    c_code = numeric_column(raw, ["证券代码"])
    c_date = date_column(raw)
    c_close = numeric_column(raw, ["收盘价"])
    fetch_time = datetime.now().isoformat(timespec="seconds")
    result = existing.copy()
    for security, asset_code in CREDIT_INDEX_ASSETS.items():
        selected = raw[raw[c_code].astype(str).str.upper().eq(security)].copy()
        old = result[result["asset_code"].astype(str).eq(asset_code)].copy()
        meta = asset_meta(result, asset_code)
        fresh = pd.DataFrame(
            {
                "date": clean_date(selected[c_date]),
                "close": pd.to_numeric(selected[c_close], errors="coerce"),
                **meta,
                "asset_code": asset_code,
                "code": asset_code,
                "ts_code": security,
                "data_source": "ifind_index_data_incremental",
                "fetch_time": fetch_time,
            }
        ).dropna(subset=["date", "close"])
        merged = upsert(old, fresh, ["asset_code", "date"])
        result = pd.concat([result[~result["asset_code"].astype(str).eq(asset_code)], fit_asset_schema(merged, result.columns)], ignore_index=True)
        status.append(
            {
                "dataset": f"asset:{asset_code}",
                "source": "ifind:index_data",
                "end": fresh["date"].max() if not fresh.empty else None,
                "rows_added_or_refreshed": len(fresh),
                "status": "ok" if not fresh.empty else "empty",
            }
        )
    return result


def refresh_nanhua_assets(
    existing: pd.DataFrame,
    cache_dir: Path,
    timeout: int,
    latest: pd.Timestamp,
    status: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    date_range = f"2026-05-15至{latest:%Y-%m-%d}"
    specs = {
        "alt_commodity": (f"南华商品指数每日收盘点位，{date_range}", ["商品指数"]),
        "alt_nonferrous": (f"南华有色金属指数每日收盘点位，{date_range}", ["有色金属指数"]),
        "alt_chemical": (f"指标精确为南华期货能化指数（代码NH0500.NHF）每日收盘点位，{date_range}，不要南华能源指数", ["能化指数"]),
    }
    result = existing.copy()
    series: dict[str, pd.DataFrame] = {}
    for asset_code, (query, needles) in specs.items():
        raw = ifind_query("edb", "get_edb_data", query, cache_dir, timeout)
        value = numeric_column(raw, needles)
        fresh_series = series_frame(raw, value, "close")
        series[asset_code] = fresh_series.rename(columns={"close": asset_code})
        old = result[result["asset_code"].astype(str).eq(asset_code)].copy()
        meta = asset_meta(result, asset_code)
        fresh = fresh_series.copy()
        for column, value in meta.items():
            fresh[column] = value
        fresh["asset_code"] = asset_code
        fresh["code"] = asset_code
        fresh["data_source"] = "ifind_edb_nanhua_incremental"
        fresh["fetch_time"] = datetime.now().isoformat(timespec="seconds")
        merged = upsert(old, fresh, ["asset_code", "date"])
        result = pd.concat([result[~result["asset_code"].astype(str).eq(asset_code)], fit_asset_schema(merged, result.columns)], ignore_index=True)
        status.append(
            {
                "dataset": f"asset:{asset_code}",
                "source": "ifind:edb",
                "end": fresh["date"].max(),
                "rows_added_or_refreshed": len(fresh),
                "status": "ok",
            }
        )
    return result, series


def refresh_reits_asset(
    pro: Any,
    existing: pd.DataFrame,
    latest: pd.Timestamp,
    raw_dir: Path,
    status: list[dict[str, Any]],
) -> pd.DataFrame:
    members: list[pd.DataFrame] = []
    for code in REITS:
        raw = tushare_call(
            pro,
            "fund_daily",
            ts_code=code,
            start_date="20210601",
            end_date=latest.strftime("%Y%m%d"),
        )
        raw.to_parquet(raw_dir / f"reits_fund_daily_{code.replace('.', '_')}_20210601_{latest:%Y%m%d}.parquet", index=False)
        if raw.empty:
            continue
        member = pd.DataFrame(
            {
                "date": clean_date(raw["trade_date"]),
                "member": code,
                "close": pd.to_numeric(raw["close"], errors="coerce"),
            }
        ).dropna()
        member["member_return_1d"] = member.sort_values("date")["close"].pct_change(fill_method=None)
        members.append(member)
        time.sleep(0.25)
    if not members:
        status.append({"dataset": "asset:alt_reits", "source": "tushare:first9", "status": "failed"})
        return existing
    member_all = pd.concat(members, ignore_index=True)
    daily = (
        member_all.dropna(subset=["member_return_1d"])
        .groupby("date", as_index=False)
        .agg(return_1d=("member_return_1d", "mean"), member_count=("member", "nunique"))
    )
    daily = daily[daily["member_count"].ge(5)].sort_values("date")
    daily["close"] = (1.0 + daily["return_1d"].fillna(0.0)).cumprod() * 100.0
    meta = asset_meta(existing, "alt_reits")
    fresh = daily.copy()
    for column, value in meta.items():
        fresh[column] = value
    fresh["asset_code"] = "alt_reits"
    fresh["code"] = "alt_reits"
    fresh["data_source"] = "tushare_first9_reits_equal_weight"
    fresh["fetch_time"] = datetime.now().isoformat(timespec="seconds")
    out = pd.concat(
        [existing[~existing["asset_code"].astype(str).eq("alt_reits")], fit_asset_schema(fresh, existing.columns)],
        ignore_index=True,
    )
    status.append(
        {
            "dataset": "asset:alt_reits",
            "source": "tushare:first9_equal_weight",
            "end": fresh["date"].max(),
            "rows_added_or_refreshed": len(fresh),
            "status": "ok",
        }
    )
    return out


def refresh_us_bond_asset(
    existing: pd.DataFrame,
    cache_dir: Path,
    timeout: int,
    latest: pd.Timestamp,
    status: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ifind_query(
        "edb",
        "get_edb_data",
        f"美国10年期国债收益率每日值，2026-04-01至{latest:%Y-%m-%d}，单位百分比",
        cache_dir,
        timeout,
    )
    value = numeric_column(raw, ["国债收益率", "10年"])
    fresh_yield = series_frame(raw, value, "us_10y_yield")
    old_alt = read_parquet("alt_overseas_filled_daily.parquet")
    old_yield = old_alt[["date", "us_10y_yield"]].copy() if not old_alt.empty else pd.DataFrame(columns=["date", "us_10y_yield"])
    old_yield["date"] = pd.to_datetime(old_yield["date"], errors="coerce")
    yields = upsert(old_yield, fresh_yield, ["date"]).dropna(subset=["us_10y_yield"]).sort_values("date")

    # Historical provider batches had a systematic Jan/Jul hole.  Backfill
    # only completely absent months from the primary yield series; never use
    # interpolation for a whole missing month.
    expected_months = pd.period_range("2018-01", pd.Timestamp(yields["date"].max()).to_period("M"), freq="M")
    present_months = set(pd.to_datetime(yields["date"]).dt.to_period("M"))
    missing_months = [month for month in expected_months if month not in present_months]
    backfills: list[pd.DataFrame] = []
    for month in missing_months:
        month_start = month.start_time.strftime("%Y-%m-%d")
        month_end = min(month.end_time.normalize(), pd.Timestamp(yields["date"].max())).strftime("%Y-%m-%d")
        missing_raw = ifind_query(
            "edb",
            "get_edb_data",
            f"美国10年期国债收益率每日值，{month_start}至{month_end}，单位百分比",
            cache_dir,
            timeout,
        )
        missing_value = numeric_column(missing_raw, ["国债收益率", "10年"])
        backfills.append(series_frame(missing_raw, missing_value, "us_10y_yield"))
    if backfills:
        yields = upsert(yields, pd.concat(backfills, ignore_index=True), ["date"])
        yields = yields.dropna(subset=["us_10y_yield"]).sort_values("date")
    dy = yields["us_10y_yield"].diff() / 100.0
    ret = yields["us_10y_yield"].shift(1) / 100.0 / 252.0 - 7.5 * dy
    ret = ret.where(ret.abs().le(0.20))
    meta = asset_meta(existing, "alt_us_bond")
    fresh = pd.DataFrame(
        {
            "date": yields["date"],
            **meta,
            "asset_code": "alt_us_bond",
            "code": "alt_us_bond",
            "close": (1.0 + ret.fillna(0.0)).cumprod() * 100.0,
            "return_1d": ret,
            "data_source": "ifind_us10y_curve_proxy:duration_7.5",
            "fetch_time": datetime.now().isoformat(timespec="seconds"),
        }
    )
    out = pd.concat(
        [existing[~existing["asset_code"].astype(str).eq("alt_us_bond")], fit_asset_schema(fresh, existing.columns)],
        ignore_index=True,
    )
    status.append(
        {
            "dataset": "asset:alt_us_bond",
            "source": "ifind:US10Y_curve_proxy",
            "end": fresh["date"].max(),
            "rows_added_or_refreshed": len(fresh),
            "months_backfilled": len(missing_months),
            "status": "ok",
        }
    )
    return out, yields


def latest_cb_database() -> Path:
    candidates = [
        path
        for path in (Path.home() / "Desktop").rglob("*.xlsx")
        if "可转债数据库生成" in str(path) and "转债数据库_" in path.name and path.stat().st_size > 50_000_000
    ]
    if not candidates:
        raise FileNotFoundError("No full local convertible-bond database was found.")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.stat().st_size))


def _read_cb_wide(database: Path, sheet: int) -> pd.DataFrame:
    frame = pd.read_excel(database, sheet_name=sheet)
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).set_index("date").sort_index()


def refresh_cb_assets_and_features(
    pro: Any,
    existing: pd.DataFrame,
    latest: pd.Timestamp,
    status: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    database = latest_cb_database()
    log(f"Reading point-in-time convertible-bond matrices from {database}")
    close = _read_cb_wide(database, 7)
    balance = _read_cb_wide(database, 11)
    static = pd.read_excel(database, sheet_name=6)
    market = pd.read_excel(database, sheet_name=0)

    common_codes = sorted(set(close.columns.astype(str)) & set(balance.columns.astype(str)))
    close = close[common_codes].apply(pd.to_numeric, errors="coerce")
    balance = balance.reindex(index=close.index, columns=common_codes).apply(pd.to_numeric, errors="coerce")

    # Extend the latest local database with Tushare closes.  Balances are held
    # from the last known date, so no future balance is used for the extension.
    for day in pd.bdate_range(close.index.max() + pd.Timedelta(days=1), latest):
        raw = tushare_call(pro, "cb_daily", trade_date=day.strftime("%Y%m%d"))
        if raw.empty:
            continue
        row = pd.Series(pd.to_numeric(raw["close"], errors="coerce").to_numpy(), index=raw["ts_code"].astype(str))
        close.loc[day, common_codes] = row.reindex(common_codes).to_numpy()
        balance.loc[day, common_codes] = balance.iloc[-1].reindex(common_codes).to_numpy()
        time.sleep(0.2)
    close = close.sort_index()
    balance = balance.reindex(close.index).ffill()

    code_col = next(c for c in static.columns if "转债代码" in str(c))
    size_col = next(c for c in static.columns if "发行规模" in str(c))
    issue_size = (
        pd.DataFrame(
            {
                "code": static[code_col].astype(str),
                "issue_size": pd.to_numeric(static[size_col], errors="coerce"),
            }
        )
        .dropna(subset=["issue_size"])
        .drop_duplicates("code", keep="last")
        .set_index("code")["issue_size"]
        .reindex(common_codes)
    )

    # Remove stale-price/redemption discontinuities that cannot represent a
    # tradable diversified index return.  The 25% bound is above the regular
    # exchange limit but rejects obvious matrix breaks.
    returns = close.pct_change(fill_method=None).where(lambda x: x.abs().le(0.25))
    signal_price = close.shift(1)
    signal_balance = balance.shift(1)
    groups = [(style, size, price) for style in ["bond", "balanced", "equity"] for size in ["large", "small"] for price in ["low", "high"]]
    group_returns: dict[tuple[str, str, str], list[float]] = {key: [] for key in groups}
    group_counts: dict[tuple[str, str, str], list[int]] = {key: [] for key in groups}
    group_fallback_levels: dict[tuple[str, str, str], list[int]] = {key: [] for key in groups}

    for day in close.index:
        p = signal_price.loc[day]
        b = signal_balance.loc[day]
        r = returns.loc[day]
        active = p.notna() & b.gt(0) & r.notna() & issue_size.notna()
        size_median = issue_size[active].median()
        style_masks = {
            "bond": active & p.le(110),
            "balanced": active & p.gt(110) & p.le(130),
            "equity": active & p.gt(130),
        }
        for style, size, price in groups:
            style_mask = style_masks[style].copy()
            size_mask = issue_size.ge(size_median) if size == "large" else issue_size.lt(size_median)
            style_size_mask = style_mask & size_mask
            style_size_median = p[style_size_mask].median()
            fine_mask = style_size_mask & (p.le(style_size_median) if price == "low" else p.gt(style_size_median))

            # Early in the sample the 12-way split can leave only one or two
            # bonds in a cell.  Use the narrowest point-in-time group with at
            # least eight names instead of publishing a fragile single-name
            # pseudo-index.  The fallback hierarchy never uses future members.
            candidates = [fine_mask, style_size_mask, style_mask, active]
            fallback_level = 3
            valid = active & r.notna()
            for level, candidate in enumerate(candidates):
                candidate_valid = candidate & r.notna()
                if int(candidate_valid.sum()) >= 8:
                    valid = candidate_valid
                    fallback_level = level
                    break
            weights = b[valid].clip(lower=0)
            if weights.sum() > 0:
                weights = weights.clip(upper=weights.sum() * 0.10)
            value = float(np.average(r[valid], weights=weights)) if valid.any() and weights.sum() > 0 else np.nan
            group_returns[(style, size, price)].append(value)
            group_counts[(style, size, price)].append(int(valid.sum()))
            group_fallback_levels[(style, size, price)].append(fallback_level)

    frames: list[pd.DataFrame] = []
    for key in groups:
        style, size, price = key
        asset_code = f"cb_{style}_{size}_{price}"
        ret = pd.Series(group_returns[key], index=close.index, dtype=float)
        count = pd.Series(group_counts[key], index=close.index, dtype=int)
        first = ret.first_valid_index()
        if first is None:
            continue
        ret = ret.loc[first:]
        count = count.loc[first:]
        meta = asset_meta(existing, asset_code)
        frame = pd.DataFrame(
            {
                "date": ret.index,
                **meta,
                "asset_code": asset_code,
                "code": asset_code,
                "asset_name": "可转债" + CB_STYLE_NAMES[style] + CB_STYLE_NAMES[size] + CB_STYLE_NAMES[price],
                "asset_class": "convertible_bonds",
                "module": "convertible_bonds",
                "sub_module": asset_code,
                "close": (1.0 + ret.fillna(0.0)).cumprod() * 100.0,
                "return_1d": ret.to_numpy(),
                "data_source": "local_cb_database_pit_lagged_balance",
                "fetch_time": datetime.now().isoformat(timespec="seconds"),
                "member_count": count.to_numpy(),
            }
        )
        frames.append(frame)
    styles = pd.concat(frames, ignore_index=True, sort=False)
    cb_codes = styles["asset_code"].unique().tolist()
    out = pd.concat(
        [existing[~existing["asset_code"].astype(str).isin(cb_codes)], fit_asset_schema(styles, existing.columns)],
        ignore_index=True,
    )

    market = market.rename(columns={market.columns[0]: "date"})
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.dropna(subset=["date"]).sort_values("date")
    feature = pd.DataFrame({"date": market["date"]})
    feature["cb_price_median"] = pd.to_numeric(market[numeric_column(market, ["绝对价格中位数"])], errors="coerce")
    feature["cb_premium_median"] = pd.to_numeric(market[numeric_column(market, ["转股溢价率中位数"])], errors="coerce")
    feature["cb_parity_median"] = pd.to_numeric(market[numeric_column(market, ["平价中位数"], ["溢价"])], errors="coerce")
    feature["cb_ytm_median"] = pd.to_numeric(market[numeric_column(market, ["YTM中位数"])], errors="coerce")
    feature["cb_iv_median"] = pd.to_numeric(market[numeric_column(market, ["隐含波动率"], ["平滑", "分位"])], errors="coerce")
    feature["cb_amount_sum"] = pd.to_numeric(market[numeric_column(market, ["市场成交额"])], errors="coerce") * 10000.0
    balance_sum = balance.sum(axis=1, min_count=1)
    count = balance.gt(0).sum(axis=1)
    price_median = close.median(axis=1, skipna=True)
    price_avg = close.mean(axis=1, skipna=True)
    supplement = pd.DataFrame(
        {
            "date": close.index,
            "cb_count": count.to_numpy(),
            "cb_balance_sum": balance_sum.to_numpy(),
            "cb_price_avg": price_avg.to_numpy(),
            "cb_price_median_db": price_median.to_numpy(),
            "low_price_cb_ratio": close.lt(100).sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan),
            "high_price_cb_ratio": close.gt(130).sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan),
        }
    ).reset_index(drop=True)
    feature = supplement.merge(feature, on="date", how="left")
    feature["cb_price_median"] = feature["cb_price_median"].combine_first(feature["cb_price_median_db"])
    feature = feature.drop(columns=["cb_price_median_db"])
    feature["source"] = f"local_cb_database:{database.name};tushare_close_extension"

    status.append(
        {
            "dataset": "asset:convertible_bond_styles",
            "source": "local_cb_database_pit",
            "end": styles["date"].max(),
            "rows_added_or_refreshed": len(styles),
            "fallback_rate": float(
                np.mean([level > 0 for levels in group_fallback_levels.values() for level in levels])
            ),
            "status": "ok",
        }
    )
    status.append(
        {
            "dataset": "feature:convertible_bond_market",
            "source": "local_cb_database_pit",
            "end": feature["date"].max(),
            "rows_added_or_refreshed": len(feature),
            "status": "ok",
        }
    )
    return out, feature, database


def refresh_rate_and_credit_sources(cache_dir: Path, timeout: int, latest: pd.Timestamp, dry_run: bool, status: list[dict[str, Any]]) -> None:
    start, end = "2026-05-15", latest.strftime("%Y-%m-%d")
    specs = {
        "treasury": ("中债国债到期收益率", [1, 3, 5, 10, 30]),
        "policy_bank": ("中债国开债到期收益率", [1, 3, 5, 10, 30]),
        "mtb_aaa": ("中债中短期票据收益率曲线(AAA)", [1, 3, 5, 10]),
        # The provider currently has no stable 5Y field for this curve.  The
        # strategy uses 1Y/3Y for the bank-bond proxy, so do not fabricate 5Y.
        "bank_bond_aaa": ("中债商业银行普通债收益率曲线(AAA)", [1, 3, 10]),
    }
    curves: dict[str, pd.DataFrame] = {}
    for key, (indicator, tenors) in specs.items():
        parts: list[pd.DataFrame] = []
        if key in {"mtb_aaa", "bank_bond_aaa"}:
            tenor_text = "、".join(f"{tenor}年" for tenor in tenors)
            query = f"{indicator}{tenor_text}，{start}至{end}，日频"
            try:
                raw = ifind_query("edb", "get_edb_data", query, cache_dir, timeout)
            except RuntimeError as exc:
                raise RuntimeError(f"Curve refresh failed for {key}: {exc}") from exc
            dcol = date_column(raw)
            for tenor in tenors:
                candidates = [
                    c
                    for c in raw.columns
                    if c != dcol and f":{tenor}年" in str(c) and "到期收益率" in str(c)
                ]
                if not candidates:
                    # iFinD occasionally omits one field from a multi-indicator
                    # request. Retry the missing tenor as a single indicator so
                    # a provider-side truncation cannot silently become a gap.
                    single_query = f"{indicator}:{tenor}年，{start}至{end}，日频"
                    single_raw = ifind_query("edb", "get_edb_data", single_query, cache_dir, timeout)
                    single_dcol = date_column(single_raw)
                    single_candidates = [
                        c
                        for c in single_raw.columns
                        if c != single_dcol and "到期收益率" in str(c)
                    ]
                    if not single_candidates:
                        raise RuntimeError(
                            f"Curve refresh omitted {key} {tenor}Y after exact retry: "
                            f"batch={list(raw.columns)}, single={list(single_raw.columns)}"
                        )
                    parts.append(series_frame(single_raw, str(single_candidates[0]), f"y{tenor}"))
                else:
                    parts.append(series_frame(raw, str(candidates[0]), f"y{tenor}"))
        else:
            for tenor in tenors:
                query = f"{indicator}:{tenor}年，{start}至{end}，日频"
                try:
                    raw = ifind_query("edb", "get_edb_data", query, cache_dir, timeout)
                except RuntimeError as exc:
                    raise RuntimeError(f"Curve refresh failed for {key} {tenor}Y: {exc}") from exc
                dcol = date_column(raw)
                value = next(c for c in raw.columns if c != dcol)
                parts.append(series_frame(raw, str(value), f"y{tenor}"))
        frame = parts[0]
        for part in parts[1:]:
            frame = frame.merge(part, on="date", how="outer")
        curves[key] = frame.sort_values("date").assign(curve_key=key)

    for rating_key, rating in [("corp_aap", "AA+"), ("corp_aa", "AA")]:
        parts = []
        for tenor in [1, 3]:
            query = f"中债企业债到期收益率({rating}):{tenor}年，{start}至{end}，日频，严格按{rating}评级"
            raw = ifind_query("edb", "get_edb_data", query, cache_dir, timeout)
            dcol = date_column(raw)
            value = next(c for c in raw.columns if c != dcol)
            part = series_frame(raw, str(value), f"y{tenor}")
            parts.append(part)
        curves[rating_key] = parts[0].merge(parts[1], on="date", how="outer").assign(curve_key=rating_key)

    long_path = PROCESSED / "credit_yield_curves_daily_v42_long.parquet"
    long_old = pd.read_parquet(long_path)
    long_new = pd.concat([curves[key] for key in ["treasury", "policy_bank", "mtb_aaa", "bank_bond_aaa"]], ignore_index=True, sort=False)
    long = upsert(long_old, long_new, ["date", "curve_key"])
    write_parquet(long, long_path.name, dry_run)

    curve_names = {
        "treasury": "中债国债收益率曲线",
        "policy_bank": "中债国开债收益率曲线",
        "mtb_aaa": "中债中短期票据收益率曲线(AAA)",
        "bank_bond_aaa": "中债商业银行普通债收益率曲线(AAA)",
    }
    raw_rows = []
    for key, frame in curves.items():
        if key not in curve_names:
            continue
        tmp = pd.DataFrame({"曲线名称": curve_names[key], "日期": frame["date"]})
        for tenor, label in [(1, "1年"), (3, "3年"), (5, "5年"), (10, "10年"), (30, "30年")]:
            tmp[label] = frame.get(f"y{tenor}")
        raw_rows.append(tmp)
    raw_old = read_parquet("raw_chinabond_yield_curves.parquet")
    raw_curve = upsert(raw_old, pd.concat(raw_rows, ignore_index=True), ["曲线名称", "日期"])
    write_parquet(raw_curve, "raw_chinabond_yield_curves.parquet", dry_run)

    merged = curves["treasury"].drop(columns="curve_key").rename(columns={f"y{x}": f"treasury_y{x}" for x in [1, 3, 5, 10, 30]})
    for key, prefix in [("mtb_aaa", "mtb_aaa"), ("bank_bond_aaa", "bank_bond_aaa"), ("corp_aap", "corp_aap"), ("corp_aa", "corp_aa")]:
        tmp = curves[key].drop(columns="curve_key", errors="ignore").rename(columns={f"y{x}": f"{prefix}_y{x}" for x in [1, 3, 5, 10, 30]})
        merged = merged.merge(tmp, on="date", how="outer")
    spread = pd.DataFrame({"date": merged["date"]})
    for prefix in ["treasury", "mtb_aaa", "bank_bond_aaa"]:
        for tenor in [1, 3, 5, 10]:
            spread[f"{prefix}_y{tenor}"] = merged.get(f"{prefix}_y{tenor}")
    spread["credit_spread_aaa_1_3y"] = (
        merged[["mtb_aaa_y1", "mtb_aaa_y3"]].mean(axis=1) - merged[["treasury_y1", "treasury_y3"]].mean(axis=1)
    )
    spread["credit_spread_aa_plus_1_3y"] = (
        merged[["corp_aap_y1", "corp_aap_y3"]].mean(axis=1) - merged[["treasury_y1", "treasury_y3"]].mean(axis=1)
    )
    spread["credit_spread_aa_1_3y"] = (
        merged[["corp_aa_y1", "corp_aa_y3"]].mean(axis=1) - merged[["treasury_y1", "treasury_y3"]].mean(axis=1)
    )
    spread["credit_spread_aaa_3_5y"] = (
        merged[["mtb_aaa_y3", "mtb_aaa_y5"]].mean(axis=1) - merged[["treasury_y3", "treasury_y5"]].mean(axis=1)
    )
    spread["credit_spread_proxy_aaa_mtb_1_3y_vs_treasury"] = spread["credit_spread_aaa_1_3y"]
    spread["credit_spread_proxy_aaa_mtb_3_5y_vs_treasury"] = spread["credit_spread_aaa_3_5y"]
    spread["credit_spread_proxy_aaa_bank_1_3y_vs_treasury"] = (
        merged[["bank_bond_aaa_y1", "bank_bond_aaa_y3"]].mean(axis=1) - merged[["treasury_y1", "treasury_y3"]].mean(axis=1)
    )
    spread["credit_spread_proxy_available"] = spread["credit_spread_aaa_1_3y"].notna()
    spread["source"] = "ifind_edb_chinabond_plus_enterprise_rating_proxy"
    spread["strict_missing_reason"] = "AA+/AA使用银行间企业债评级曲线，非中票同券种口径"
    old_spread = read_parquet("credit_spreads_proxy_daily_v42.parquet")
    write_parquet(upsert(old_spread, spread, ["date"]), "credit_spreads_proxy_daily_v42.parquet", dry_run)

    bank = curves["bank_bond_aaa"][["date", "y1"]].rename(columns={"y1": "aaa_cd_1y_proxy"})
    shibor = read_parquet("shibor_rates_daily_v42.parquet")
    cd = bank.merge(shibor[["date", "shibor_3m", "shibor_1y"]], on="date", how="left")
    cd["aaa_cd_proxy_source"] = "ifind_chinabond_commercial_bank_aaa_1y"
    cd["shibor_3m_proxy"] = cd["shibor_3m"]
    cd["shibor_1y_proxy"] = cd["shibor_1y"]
    cd["aaa_cd_1y_final"] = cd["aaa_cd_1y_proxy"].combine_first(cd["shibor_1y_proxy"])
    cd["source"] = "ifind_edb_proxy"
    cd["note"] = "AAA同业存单1Y仍以商业银行普通债AAA 1Y代理。"
    cd = cd.drop(columns=["shibor_3m", "shibor_1y"])
    old_cd = read_parquet("aaa_cd_1y_proxy_daily.parquet")
    write_parquet(upsert(old_cd, cd, ["date"]), "aaa_cd_1y_proxy_daily.parquet", dry_run)

    for key, frame in curves.items():
        status.append(
            {
                "dataset": f"curve:{key}",
                "source": "ifind:edb",
                "end": frame["date"].max(),
                "rows_added_or_refreshed": len(frame),
                "status": "ok",
            }
        )


def refresh_funding_sources(pro: Any, cache_dir: Path, timeout: int, latest: pd.Timestamp, dry_run: bool, status: list[dict[str, Any]]) -> None:
    shibor_raw = tushare_call(pro, "shibor", start_date="20260515", end_date=latest.strftime("%Y%m%d"))
    old_shibor = read_parquet("shibor_rates_daily_v42.parquet")
    required_shibor = {"date", "on", "1w", "3m", "1y"}
    if shibor_raw.empty or not required_shibor.issubset(shibor_raw.columns):
        shibor = pd.DataFrame(columns=["date", "shibor_on", "shibor_1w", "shibor_3m", "shibor_1y", "source"])
        shibor_status = "not_published_no_change"
        shibor_end = pd.to_datetime(old_shibor["date"], errors="coerce").max()
    else:
        shibor = pd.DataFrame(
            {
                "date": clean_date(shibor_raw["date"]),
                "shibor_on": pd.to_numeric(shibor_raw["on"], errors="coerce"),
                "shibor_1w": pd.to_numeric(shibor_raw["1w"], errors="coerce"),
                "shibor_3m": pd.to_numeric(shibor_raw["3m"], errors="coerce"),
                "shibor_1y": pd.to_numeric(shibor_raw["1y"], errors="coerce"),
                "source": "tushare_shibor_incremental",
            }
        ).dropna(subset=["date"])
        shibor_status = "ok"
        shibor_end = shibor["date"].max()
        write_parquet(upsert(old_shibor, shibor, ["date"]), "shibor_rates_daily_v42.parquet", dry_run)

    r_raw = ifind_query(
        "edb",
        "get_edb_data",
        f"银行间质押式回购利率R007每日值，2026-05-15至{latest:%Y-%m-%d}，单位百分比，仅R007",
        cache_dir,
        timeout,
    )
    dr_raw = ifind_query(
        "edb",
        "get_edb_data",
        f"银行间存款类机构质押式回购加权利率DR007每日值，2026-05-15至{latest:%Y-%m-%d}，单位百分比，仅DR007",
        cache_dir,
        timeout,
    )
    r = series_frame(r_raw, next(c for c in r_raw.columns if c != date_column(r_raw)), "r007")
    dr = series_frame(dr_raw, next(c for c in dr_raw.columns if c != date_column(dr_raw)), "dr007")
    funding = r.merge(dr, on="date", how="outer").sort_values("date")
    funding["r007_dr007_spread"] = funding["r007"] - funding["dr007"]
    stress_raw = funding["r007_dr007_spread"].rolling(60, min_periods=20)
    funding["funding_stress_score_proxy"] = (funding["r007_dr007_spread"] - stress_raw.mean()) / stress_raw.std()
    funding["source"] = "ifind_edb_R007_DR007_incremental"
    old = read_parquet("funding_rates_daily_v42.parquet")
    write_parquet(upsert(old, funding, ["date"]), "funding_rates_daily_v42.parquet", dry_run)
    status.extend(
        [
            {"dataset": "funding:shibor", "source": "tushare", "end": shibor_end, "rows_added_or_refreshed": len(shibor), "status": shibor_status},
            {"dataset": "funding:R007_DR007", "source": "ifind:edb", "end": funding["date"].max(), "rows_added_or_refreshed": len(funding), "status": "ok"},
        ]
    )


def refresh_alt_feature_source(
    cache_dir: Path,
    timeout: int,
    us_yield: pd.DataFrame,
    nanhua: dict[str, pd.DataFrame],
    latest: pd.Timestamp,
    dry_run: bool,
    status: list[dict[str, Any]],
) -> None:
    date_range = f"2026-05-15至{latest:%Y-%m-%d}"
    specs = {
        "usd_index": (f"美元指数DXY每日收盘值，{date_range}", ["美元指数"]),
        "gold_au9999": (f"上海黄金交易所Au99.99黄金每日收盘价，{date_range}，单位元每克", ["Au9999"]),
    }
    frames = [us_yield]
    for name, (query, needles) in specs.items():
        raw = ifind_query("edb", "get_edb_data", query, cache_dir, timeout)
        frames.append(series_frame(raw, numeric_column(raw, needles), name))
    frames.append(nanhua["alt_commodity"].rename(columns={"alt_commodity": "nanhua_commodity"}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    old = read_parquet("alt_overseas_filled_daily.parquet")
    combined = upsert(old, merged, ["date"]).sort_values("date")
    for column in ["us_10y_yield", "usd_index", "nanhua_commodity", "gold_au9999"]:
        combined[f"{column}_ret_1d"] = pd.to_numeric(combined[column], errors="coerce").pct_change(fill_method=None)
    combined["source"] = combined.get("source", pd.Series(index=combined.index, dtype=object)).fillna("ifind_incremental")
    write_parquet(combined, "alt_overseas_filled_daily.parquet", dry_run)
    status.append(
        {
            "dataset": "feature:alternatives",
            "source": "ifind:edb",
            "end": merged["date"].max(),
            "rows_added_or_refreshed": len(merged),
            "status": "ok",
        }
    )


def refresh_macro_source(cache_dir: Path, timeout: int, dry_run: bool, status: list[dict[str, Any]]) -> None:
    queries = {
        "core": "中国CPI同比、PPI同比、官方制造业PMI，2026年1月至2026年6月，月频",
        "m2": "中国M2同比，2026年1月至2026年6月，月频",
        "social": "中国社会融资规模增量当月值，2026年1月至2026年6月，月频，单位亿元",
        "industrial": "中国规模以上工业增加值当月同比，2026年1月至2026年6月，月频，不要累计同比",
        "fai": "中国固定资产投资完成额累计同比，2026年1月至2026年6月，月频",
        "export": "中国出口金额人民币计价当月同比，2026年1月至2026年6月，月频",
    }
    raw = {key: ifind_query("edb", "get_edb_data", query, cache_dir, timeout) for key, query in queries.items()}

    core = raw["core"]
    month = clean_date(core[date_column(core)]) + pd.offsets.MonthEnd(0)
    new = pd.DataFrame({"month": month})
    new["cpi_yoy"] = pd.to_numeric(core[numeric_column(core, ["CPI", "同比"])], errors="coerce")
    new["ppi_yoy"] = pd.to_numeric(core[numeric_column(core, ["PPI", "同比"])], errors="coerce")
    pmi_col = numeric_column(core, ["PMI"])
    new["manufacturing_pmi"] = pd.to_numeric(core[pmi_col], errors="coerce")

    additions = {
        "m2": ("m2_yoy", ["M2", "同比"], 1.0),
        "social": ("social_financing_new", ["社会融资规模增量"], 1.0 / 1e8),
        "industrial": ("industrial_production_yoy", ["工业增加值", "同比"], 1.0),
        "fai": ("fixed_asset_investment_yoy", ["固定资产投资", "同比"], 1.0),
        "export": ("export_yoy", ["出口", "同比"], 1.0),
    }
    for key, (output, needles, scale) in additions.items():
        frame = raw[key]
        part = pd.DataFrame(
            {
                "month": clean_date(frame[date_column(frame)]) + pd.offsets.MonthEnd(0),
                output: pd.to_numeric(frame[numeric_column(frame, needles)], errors="coerce") * scale,
            }
        )
        new = new.merge(part, on="month", how="outer")
    new["effective_date_conservative"] = new["month"] + pd.offsets.Day(20)
    new["source"] = "ifind_edb_release_lagged"
    old = read_parquet("macro_factors_monthly_v42.parquet")
    combined = upsert(old, new, ["month"])
    write_parquet(combined, "macro_factors_monthly_v42.parquet", dry_run)
    status.append(
        {
            "dataset": "macro:monthly",
            "source": "ifind:edb_release_lagged",
            "end": new["month"].max(),
            "rows_added_or_refreshed": len(new),
            "status": "ok",
        }
    )


def refresh_equity_sources(pro: Any, latest: pd.Timestamp, dry_run: bool, status: list[dict[str, Any]]) -> None:
    index_specs = {
        "csi1000": "000852.SH",
        "dividend": "000922.CSI",
        "csi_convertible_bond": "000832.CSI",
    }
    rows = []
    for name, code in index_specs.items():
        quote = tushare_call(pro, "index_daily", ts_code=code, start_date="20260515", end_date=latest.strftime("%Y%m%d"))
        basic = tushare_call(pro, "index_dailybasic", ts_code=code, start_date="20260515", end_date=latest.strftime("%Y%m%d"))
        if quote.empty:
            continue
        quote["date"] = clean_date(quote["trade_date"])
        if not basic.empty:
            basic["date"] = clean_date(basic["trade_date"])
            keep = [c for c in ["date", "pe", "pe_ttm", "pb"] if c in basic.columns]
            quote = quote.merge(basic[keep], on="date", how="left")
        pe = numeric_series(quote, "pe_ttm", "pe")
        rows.append(
            pd.DataFrame(
                {
                    "date": quote["date"],
                    "index_name": name,
                    "wind_code": code,
                    "pe": pe,
                    "pb": numeric_series(quote, "pb"),
                    "close": pd.to_numeric(quote["close"], errors="coerce"),
                    "amount": numeric_series(quote, "amount"),
                    "source": "tushare_index_incremental",
                }
            )
        )
        time.sleep(0.25)
    if rows:
        old = read_parquet("index_valuation_filled_daily.parquet")
        new = pd.concat(rows, ignore_index=True)
        write_parquet(upsert(old, new, ["date", "index_name"]), "index_valuation_filled_daily.parquet", dry_run)
        status.append({"dataset": "equity:index_valuation", "source": "tushare", "end": new["date"].max(), "rows_added_or_refreshed": len(new), "status": "ok"})

    sw_old = read_parquet("sw_industry_returns_daily_v42.parquet")
    if sw_old.empty:
        return
    codes = sw_old["ts_code"].dropna().astype(str).drop_duplicates().tolist()
    sw_frames = []
    for code in codes:
        frame = tushare_call(pro, "sw_daily", ts_code=code, start_date="20260515", end_date=latest.strftime("%Y%m%d"))
        if frame.empty:
            continue
        frame["date"] = clean_date(frame["trade_date"])
        if "pct_change" in frame.columns:
            frame["pct_chg"] = pd.to_numeric(frame["pct_change"], errors="coerce")
        sw_frames.append(frame)
        time.sleep(0.15)
    if not sw_frames:
        return
    sw_new = pd.concat(sw_frames, ignore_index=True, sort=False)
    sw = upsert(sw_old, sw_new, ["ts_code", "date"])
    sw = sw.sort_values(["ts_code", "date"])
    sw["pct_chg"] = pd.to_numeric(sw.get("pct_chg"), errors="coerce")
    sw["ret_5d"] = sw.groupby("ts_code")["close"].pct_change(5, fill_method=None)
    sw["ret_20d"] = sw.groupby("ts_code")["close"].pct_change(20, fill_method=None)
    write_parquet(sw, "sw_industry_returns_daily_v42.parquet", dry_run)
    breadth = (
        sw.groupby("date", as_index=False)
        .agg(
            sw_l1_count=("ts_code", "nunique"),
            sw_l1_positive_1d_ratio=("pct_chg", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            sw_l1_positive_5d_ratio=("ret_5d", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            sw_l1_positive_20d_ratio=("ret_20d", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            sw_l1_median_1d_ret=("pct_chg", "median"),
        )
    )
    breadth["source"] = "tushare_sw_daily_SW2021_L1_incremental"
    write_parquet(breadth, "equity_industry_breadth_daily_v42.parquet", dry_run)
    status.append({"dataset": "equity:industry_breadth", "source": "tushare:sw_daily", "end": breadth["date"].max(), "rows_added_or_refreshed": len(sw_new), "status": "ok"})


def recompute_asset_outputs(daily: pd.DataFrame, complete_month: pd.Timestamp, dry_run: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    # The research calendar is a trading-day calendar.  Some incremental index
    # responses repeat Friday's close over weekends; retaining those rows would
    # silently shorten 5/20/60-day labels and contaminate rolling windows.
    daily = daily.loc[daily["date"].dt.dayofweek.lt(5)].copy()
    daily = daily.dropna(subset=["date", "asset_code", "close"]).drop_duplicates(["asset_code", "date"], keep="last")
    daily = daily.sort_values(["asset_code", "date"]).reset_index(drop=True)
    daily["return_1d"] = daily.groupby("asset_code")["close"].pct_change(fill_method=None)
    previous_source = daily.groupby("asset_code")["data_source"].shift()
    source_switch = daily["data_source"].ne(previous_source) & previous_source.notna()
    # A source handover may use a different index base even when the economic
    # series is continuous.  Never turn that rebasing jump into investment PnL.
    daily.loc[source_switch | daily["return_1d"].abs().gt(0.80), "return_1d"] = np.nan
    daily["data_quality_flag"] = np.where(
        source_switch,
        "source_switch_return_withheld",
        "trading_day_observation",
    )

    monthly_rows = []
    for _, group in daily.groupby("asset_code", sort=False):
        month = group.set_index("date").resample("ME").last().dropna(subset=["close"]).reset_index()
        month = month[month["date"].le(complete_month)].copy()
        month["return_1m"] = month["close"].pct_change(fill_method=None)
        monthly_rows.append(month)
    monthly = pd.concat(monthly_rows, ignore_index=True, sort=False)
    old_monthly = read_parquet("asset_returns_monthly.parquet")
    keep = list(old_monthly.columns) if not old_monthly.empty else ["date", "asset_code", "code", "asset_name", "asset_class", "module", "sub_module", "close", "return_1m", "data_source", "fetch_time"]
    for column in keep:
        if column not in monthly.columns:
            monthly[column] = np.nan
    monthly = monthly[keep].sort_values(["asset_code", "date"]).reset_index(drop=True)

    if not dry_run:
        daily.to_parquet(PROCESSED / "asset_returns_daily.parquet", index=False)
        monthly.to_parquet(PROCESSED / "asset_returns_monthly.parquet", index=False)
        sys.path.insert(0, str(ROOT))
        from src.data_prep_pipeline import build_feature_panels

        feature_daily, feature_monthly = build_feature_panels(daily, monthly)
        feature_daily.to_parquet(PROCESSED / "feature_panel_daily.parquet", index=False)
        feature_monthly.to_parquet(PROCESSED / "feature_panel_monthly.parquet", index=False)
    return daily, monthly


def rebuild_asset_universe_from_daily(daily: pd.DataFrame, dry_run: bool) -> pd.DataFrame:
    """Materialize the actual dynamic 38-asset universe as the master table.

    Convertible-bond style buckets are generated from the point-in-time CB
    database and therefore do not appear in ``asset_universe.yaml``.  The old
    master table described only the static 26 configured assets, which made a
    valid 38-asset backtest hard to reproduce.  This table records every asset
    that actually reached the return matrix, including its coverage and source.
    """
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "asset_code"]).sort_values(["asset_code", "date"])
    latest = frame.groupby("asset_code", as_index=False).tail(1).set_index("asset_code")
    coverage = frame.groupby("asset_code", as_index=True).agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_obs=("date", "size"),
    )
    old = read_parquet("asset_universe.parquet")
    old = old.drop_duplicates("asset_code", keep="last").set_index("asset_code") if not old.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for asset_code in coverage.index.astype(str):
        row = latest.loc[asset_code]
        prior = old.loc[asset_code] if not old.empty and asset_code in old.index else pd.Series(dtype=object)
        module = row.get("module", prior.get("module", ""))
        source = row.get("data_source", prior.get("data_source", ""))
        is_cb_bucket = str(module) == "convertible_bonds" and str(asset_code).startswith("cb_")
        note = prior.get("note", "")
        if is_cb_bucket:
            note = (
                "点时可转债数据库按前一交易日价格与余额构建的风格桶；"
                "债性/平衡/股性 × 大盘/小盘 × 低价/高价，不是可直接交易的单一ETF。"
            )
        rows.append(
            {
                "asset_code": asset_code,
                "code": row.get("code", prior.get("code", asset_code)),
                "name": row.get("asset_name", prior.get("name", asset_code)),
                "module": module,
                "source_type": prior.get("source_type", "derived_style_bucket" if is_cb_bucket else "observed_series"),
                "ts_code": row.get("ts_code", prior.get("ts_code", None)),
                "data_source": source,
                "note": note,
                "fetch_time": row.get("fetch_time", prior.get("fetch_time", None)),
                "status": "active",
                "start_date": coverage.at[asset_code, "start_date"],
                "end_date": coverage.at[asset_code, "end_date"],
                "n_obs": int(coverage.at[asset_code, "n_obs"]),
            }
        )
    universe = pd.DataFrame(rows).sort_values(["module", "asset_code"]).reset_index(drop=True)
    if not dry_run:
        universe.to_parquet(PROCESSED / "asset_universe.parquet", index=False)
    return universe


def write_refresh_report(
    status: list[dict[str, Any]],
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    latest: pd.Timestamp,
    complete_month: pd.Timestamp,
    cb_database: Path | None,
    backup: Path,
    dry_run: bool,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(status)
    if not table.empty and "end" in table:
        table["end"] = pd.to_datetime(table["end"], errors="coerce")
    if not dry_run:
        table.to_csv(REPORTS / "latest_data_refresh_status.csv", index=False, encoding="utf-8-sig")

    coverage = (
        daily.groupby(["module", "asset_code"], as_index=False)
        .agg(start=("date", "min"), end=("date", "max"), rows=("date", "size"), missing_return=("return_1d", lambda x: float(x.isna().mean())))
        .sort_values(["module", "asset_code"])
    )
    if not dry_run:
        coverage.to_csv(REPORTS / "latest_asset_coverage.csv", index=False, encoding="utf-8-sig")

    stale = coverage[coverage["end"].lt(latest - pd.Timedelta(days=5))]
    lines = [
        "# 最新数据刷新报告",
        "",
        f"- 执行时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 最新完整交易日：{latest.date()}",
        f"- 正式月频截止日：{complete_month.date()}",
        f"- 日频资产数：{daily['asset_code'].nunique()}",
        f"- 月频资产数：{monthly['asset_code'].nunique()}",
        f"- 转债数据库：{cb_database if cb_database else '未重建'}",
        f"- 覆盖前备份：{backup}",
        "",
        "## 口径",
        "",
        "- 7月未结束，因此7月日频只进入最新状态卡，不进入月频收益回测。",
        "- 宏观数据按保守发布日期生效；报告期数据不会在公布前进入状态卡。",
        "- 转债风格指数按前一交易日价格分组、前一交易日余额加权，消除固定存续样本的幸存者偏差。",
        "- iFinD原始响应保存在 data_raw/ifind，Tushare增量保存在 data_raw/tushare/latest_refresh。",
        "",
        "## 仍有滞后但不阻塞月频回测",
        "",
    ]
    if stale.empty:
        lines.append("- 无超过5日的资产级滞后。")
    else:
        for row in stale.itertuples(index=False):
            lines.append(f"- {row.asset_code}: {pd.Timestamp(row.end).date()}")
    if not dry_run:
        (REPORTS / "latest_data_refresh_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the V1 data stack from Tushare, iFinD, and the local CB database.")
    parser.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--ifind-timeout", type=int, default=120)
    parser.add_argument("--skip-cb-rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of).normalize()
    pro = ts.pro_api(ts.get_token())
    latest = latest_complete_trade_date(pro, as_of)
    complete_month = completed_month_end(latest)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = RAW / "ifind" / f"refresh_{as_of:%Y%m%d}"
    tushare_dir = RAW / "tushare" / "latest_refresh"
    tushare_dir.mkdir(parents=True, exist_ok=True)
    status: list[dict[str, Any]] = []
    log(f"Latest completed close={latest.date()}, completed month={complete_month.date()}")

    backup_names = [
        "asset_returns_daily.parquet",
        "asset_returns_monthly.parquet",
        "feature_panel_daily.parquet",
        "feature_panel_monthly.parquet",
        "raw_chinabond_yield_curves.parquet",
        "credit_yield_curves_daily_v42_long.parquet",
        "credit_spreads_proxy_daily_v42.parquet",
        "funding_rates_daily_v42.parquet",
        "shibor_rates_daily_v42.parquet",
        "aaa_cd_1y_proxy_daily.parquet",
        "macro_factors_monthly_v42.parquet",
        "index_valuation_filled_daily.parquet",
        "sw_industry_returns_daily_v42.parquet",
        "equity_industry_breadth_daily_v42.parquet",
        "alt_overseas_filled_daily.parquet",
        "existing_cb_market_features_daily.parquet",
    ]
    backup = backup_files(backup_names, stamp, args.dry_run)

    config = yaml.safe_load((ROOT / "config" / "asset_universe.yaml").read_text(encoding="utf-8")) or {}
    daily = read_parquet("asset_returns_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    log("Refreshing configured Tushare assets")
    daily = refresh_tushare_assets(pro, daily, config, latest, tushare_dir, status)
    log("Refreshing iFinD credit index assets")
    daily = refresh_credit_index_assets(daily, cache_dir, args.ifind_timeout, latest, status)
    log("Refreshing Nanhua alternative indices")
    daily, nanhua = refresh_nanhua_assets(daily, cache_dir, args.ifind_timeout, latest, status)
    log("Refreshing public REITs equal-weight proxy")
    daily = refresh_reits_asset(pro, daily, latest, tushare_dir, status)
    log("Refreshing US Treasury curve proxy")
    daily, us_yield = refresh_us_bond_asset(daily, cache_dir, args.ifind_timeout, latest, status)

    cb_database: Path | None = None
    if not args.skip_cb_rebuild:
        log("Rebuilding point-in-time convertible-bond style assets")
        daily, cb_features, cb_database = refresh_cb_assets_and_features(pro, daily, latest, status)
        if not args.dry_run:
            cb_features.to_parquet(PROCESSED / "existing_cb_market_features_daily.parquet", index=False)

    log("Refreshing funding and Shibor")
    refresh_funding_sources(pro, cache_dir, args.ifind_timeout, latest, args.dry_run, status)
    log("Refreshing rates and credit curves")
    refresh_rate_and_credit_sources(cache_dir, args.ifind_timeout, latest, args.dry_run, status)
    log("Refreshing equity valuation and industry breadth")
    refresh_equity_sources(pro, latest, args.dry_run, status)
    log("Refreshing alternative-asset feature sources")
    refresh_alt_feature_source(cache_dir, args.ifind_timeout, us_yield, nanhua, latest, args.dry_run, status)
    log("Refreshing release-lagged macro data")
    refresh_macro_source(cache_dir, args.ifind_timeout, args.dry_run, status)

    log("Recomputing daily and completed-month asset returns")
    daily, monthly = recompute_asset_outputs(daily, complete_month, args.dry_run)
    universe = rebuild_asset_universe_from_daily(daily, args.dry_run)
    write_refresh_report(status, daily, monthly, latest, complete_month, cb_database, backup, args.dry_run)
    log("Latest-data refresh complete")
    print(
        json.dumps(
            {
                "latest_trade_date": str(latest.date()),
                "completed_month_end": str(complete_month.date()),
                "daily_assets": int(daily["asset_code"].nunique()),
                "universe_assets": int(len(universe)),
                "daily_rows": int(len(daily)),
                "monthly_rows": int(len(monthly)),
                "status_rows": len(status),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
