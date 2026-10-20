from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep_pipeline import build_monthly_returns


IFIND_DIR = Path(r"C:\Users\lenovo\.agents\skills\ifind-finance-data")
IFIND_CALL = IFIND_DIR / "call-node.js"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
RAW = ROOT / "data_raw" / "ifind" / "chinabond_history"

SPECS = [
    {
        "asset_code": "rate_1_3y",
        "asset_name": "利率债1-3Y",
        "module": "rates",
        "ifind_code": "CBA00621.CB",
        "index_name": "中债国债总指数(1-3年)财富指数",
        "index_launch_date": "2002-01-04",
        "proxy_note": "直接分期限国债财富指数，不是ETF价格。",
    },
    {
        "asset_code": "rate_3_5y",
        "asset_name": "利率债3-5Y",
        "module": "rates",
        "ifind_code": "CBA00631.CB",
        "index_name": "中债国债总指数(3-5年)财富指数",
        "index_launch_date": "2002-01-04",
        "proxy_note": "直接分期限国债财富指数，不是ETF价格。",
    },
    {
        "asset_code": "rate_7_10y",
        "asset_name": "利率债7-10Y",
        "module": "rates",
        "ifind_code": "CBA00651.CB",
        "index_name": "中债国债总指数(7-10年)财富指数",
        "index_launch_date": "2002-01-04",
        "proxy_note": "直接分期限国债财富指数，不是ETF价格。",
    },
    {
        "asset_code": "rate_30y",
        "asset_name": "超长利率债30Y",
        "module": "rates",
        "ifind_code": "CBA00661.CB",
        "index_name": "中债国债总指数(10年以上)财富指数",
        "index_launch_date": "2002-01-04",
        "proxy_note": "2018起使用10年以上国债财富指数作为30Y可回测代理，不等同于纯30Y指数或ETF。",
    },
    {
        "asset_code": "credit_ind_aaa",
        "asset_name": "产业债AAA",
        "module": "credit",
        "ifind_code": "CBA04201.CB",
        "index_name": "中债企业债AAA指数(总值)财富指数",
        "index_launch_date": "2013-08-30",
        "proxy_note": "企业债AAA财富指数作为产业债AAA代理。",
    },
    {
        "asset_code": "credit_ind_aa_plus",
        "asset_name": "产业债AA+",
        "module": "credit",
        "ifind_code": "CBA04101.CB",
        "index_name": "中债企业债AA+指数(总值)财富指数",
        "index_launch_date": "2013-08-30",
        "proxy_note": "企业债AA+财富指数作为产业债AA+代理。",
    },
    {
        "asset_code": "credit_ind_aa",
        "asset_name": "产业债AA",
        "module": "credit",
        "ifind_code": "CBA04001.CB",
        "index_name": "中债企业债AA指数(总值)财富指数",
        "index_launch_date": "2013-08-30",
        "proxy_note": "企业债AA财富指数作为产业债AA代理。",
    },
]


def ifind_query(query: str, timeout: int = 180) -> tuple[dict[str, Any], str]:
    if not IFIND_CALL.exists():
        raise FileNotFoundError(f"Missing iFinD connector: {IFIND_CALL}")
    javascript = (
        "const m=require(process.argv[1]);"
        "m.call('index','index_data',{query:process.argv[2]})"
        ".then(r=>console.log(JSON.stringify(r)))"
        ".catch(e=>{console.error(e.stack||e.message);process.exit(1)});"
    )
    result = subprocess.run(
        ["node", "-e", javascript, str(IFIND_CALL), query],
        cwd=IFIND_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout)[-1200:])
    outer = json.loads(result.stdout)
    if not outer.get("ok"):
        raise RuntimeError(json.dumps(outer.get("error"), ensure_ascii=False))
    rpc = outer.get("data", {})
    content = rpc.get("result", {}).get("content", [])
    if not content:
        raise RuntimeError("iFinD returned no content")
    payload = json.loads(content[0]["text"])
    if payload.get("code") != 1:
        raise RuntimeError(f"iFinD payload error: {payload.get('msg')} {payload.get('subMsg')}")
    text = str(payload.get("data", {}).get("text", ""))
    return outer, text


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except Exception as exc:  # pragma: no cover - provider-specific encodings
            errors.append(f"{encoding}:{exc}")
    raise RuntimeError("Unable to decode iFinD CSV: " + " | ".join(errors))


def extract_best_table(text: str, raw_prefix: Path) -> tuple[pd.DataFrame, str]:
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s\\\"']+?\.csv", text)))
    candidates: list[tuple[int, pd.DataFrame, str]] = []
    for index, url in enumerate(urls, start=1):
        content = download(url)
        csv_path = raw_prefix.with_name(f"{raw_prefix.name}_{index}.csv")
        csv_path.write_bytes(content)
        frame = read_csv_bytes(content)
        candidates.append((len(frame), frame, url))
    if not candidates:
        table_lines = [line for line in text.splitlines() if line.strip().startswith("|")]
        if len(table_lines) < 3:
            raise RuntimeError("iFinD response contains neither downloadable CSV nor a markdown table")
        frame = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="|").iloc[:, 1:-1]
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.iloc[1:].copy()
        return frame, "inline_markdown"
    _, best, source_url = max(candidates, key=lambda item: item[0])
    return best, source_url


def find_column(columns: pd.Index, needles: tuple[str, ...]) -> str:
    normalized = {str(column).strip(): column for column in columns}
    for clean, original in normalized.items():
        if all(needle in clean for needle in needles):
            return str(original)
    raise KeyError(f"Cannot find column containing {needles}; columns={list(columns)}")


def normalize_index(
    frame: pd.DataFrame,
    spec: dict[str, str],
    trading_calendar: pd.DatetimeIndex,
    fetch_time: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    date_col = find_column(frame.columns, ("日期",))
    close_col = find_column(frame.columns, ("收盘价",))
    code_col = next((column for column in frame.columns if "证券代码" in column), None)
    if code_col is not None:
        frame = frame.loc[frame[code_col].astype(str).str.upper().eq(spec["ifind_code"])].copy()
    raw_date = frame[date_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    frame["date"] = pd.to_datetime(raw_date, format="%Y%m%d", errors="coerce")
    frame["close"] = pd.to_numeric(
        frame[close_col].astype(str).str.replace("\\t", "", regex=False).str.strip(), errors="coerce"
    )
    frame = frame.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    frame = frame.loc[frame["date"].isin(trading_calendar)].sort_values("date")
    if frame.empty:
        raise RuntimeError(f"No valid trading-day observations for {spec['ifind_code']}")
    frame["return_1d"] = frame["close"].pct_change(fill_method=None)
    return pd.DataFrame(
        {
            "date": frame["date"],
            "asset_code": spec["asset_code"],
            "code": spec["asset_code"],
            "asset_name": spec["asset_name"],
            "asset_class": spec["module"],
            "module": spec["module"],
            "sub_module": spec["asset_code"],
            "ts_code": spec["ifind_code"],
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": frame["close"].to_numpy(),
            "vol": np.nan,
            "amount": np.nan,
            "return_1d": frame["return_1d"].to_numpy(),
            "data_source": "ifind_chinabond_wealth_index",
            "fetch_time": fetch_time,
            "member_count": np.nan,
        }
    )


def coverage(frame: pd.DataFrame, version: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        sub = frame.loc[frame["asset_code"].astype(str).eq(spec["asset_code"])].copy()
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
        rows.append(
            {
                "version": version,
                "asset_code": spec["asset_code"],
                "asset_name": spec["asset_name"],
                "source_code": spec["ifind_code"],
                "source_index": spec["index_name"],
                "index_launch_date": spec["index_launch_date"],
                "rows": int(len(sub)),
                "start": sub["date"].min(),
                "end": sub["date"].max(),
                "return_missing_rate": float(pd.to_numeric(sub.get("return_1d"), errors="coerce").isna().mean()) if len(sub) else 1.0,
                "proxy_note": spec["proxy_note"],
            }
        )
    return pd.DataFrame(rows)


def backup_inputs(tag: str) -> Path:
    archive = PROCESSED / "archive" / f"pre_chinabond_backfill_{tag}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("asset_returns_daily.parquet", "asset_returns_monthly.parquet", "asset_universe.parquet"):
        source = PROCESSED / name
        if source.exists():
            shutil.copy2(source, archive / name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill long-history ChinaBond wealth-index proxies through iFinD.")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2026-07-15")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    daily_path = PROCESSED / "asset_returns_daily.parquet"
    universe_path = PROCESSED / "asset_universe.parquet"
    daily = pd.read_parquet(daily_path)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    calendar = pd.DatetimeIndex(
        daily.loc[daily["asset_code"].astype(str).eq("equity_broad"), "date"].dropna().sort_values().unique()
    )
    calendar = calendar[(calendar >= pd.Timestamp(args.start_date)) & (calendar <= pd.Timestamp(args.end_date))]
    if calendar.empty:
        raise RuntimeError("The equity_broad China trading calendar is empty")

    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    fetch_time = datetime.now().isoformat(timespec="seconds")
    new_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for spec in SPECS:
        prefix = RAW / f"{spec['asset_code']}_{spec['ifind_code'].replace('.', '_')}_{args.start_date}_{args.end_date}"
        query = (
            f"{spec['ifind_code']}在{args.start_date}至{args.end_date}每日收盘点位，"
            "返回证券代码、证券简称、日期和收盘价"
        )
        cached_csvs = sorted(prefix.parent.glob(f"{prefix.name}_*.csv"))
        if prefix.with_suffix(".json").exists() and cached_csvs:
            cached_tables = [(path.stat().st_size, read_csv_bytes(path.read_bytes()), path) for path in cached_csvs]
            _, table, cached_path = max(cached_tables, key=lambda item: item[0])
            source_url = f"cached:{cached_path.relative_to(ROOT)}"
        else:
            raw_json, text = ifind_query(query)
            prefix.with_suffix(".json").write_text(json.dumps(raw_json, ensure_ascii=False, indent=2), encoding="utf-8")
            table, source_url = extract_best_table(text, prefix)
        normalized = normalize_index(table, spec, calendar, fetch_time)
        new_frames.append(normalized)
        source_rows.append(
            {
                "asset_code": spec["asset_code"],
                "ifind_code": spec["ifind_code"],
                "rows": len(normalized),
                "start": normalized["date"].min(),
                "end": normalized["date"].max(),
                "download_source": source_url,
                "status": "ok",
            }
        )
        print(f"{spec['asset_code']}: {len(normalized)} rows {normalized['date'].min().date()}..{normalized['date'].max().date()}", flush=True)

    replacement = pd.concat(new_frames, ignore_index=True)
    target_codes = {spec["asset_code"] for spec in SPECS}
    before = coverage(daily, "before")
    daily_out = pd.concat(
        [daily.loc[~daily["asset_code"].astype(str).isin(target_codes)], replacement],
        ignore_index=True,
        sort=False,
    )
    for column in daily.columns:
        if column not in daily_out.columns:
            daily_out[column] = np.nan
    daily_out = daily_out[daily.columns].sort_values(["date", "asset_code"]).reset_index(drop=True)
    after = coverage(daily_out, "after")
    report = pd.concat([before, after], ignore_index=True)
    report.to_csv(REPORTS / "chinabond_long_history_backfill_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_rows).to_csv(REPORTS / "chinabond_long_history_fetch_status.csv", index=False, encoding="utf-8-sig")

    if args.apply:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = backup_inputs(tag)
        daily_out.to_parquet(daily_path, index=False)
        build_monthly_returns(daily_out).to_parquet(PROCESSED / "asset_returns_monthly.parquet", index=False)
        if universe_path.exists():
            universe = pd.read_parquet(universe_path)
            code_column = "asset_code" if "asset_code" in universe.columns else "code"
            for spec in SPECS:
                selected = universe[code_column].astype(str).eq(spec["asset_code"])
                universe.loc[selected, "ts_code"] = spec["ifind_code"]
                universe.loc[selected, "data_source"] = "ifind_chinabond_wealth_index"
                if "source_type" in universe.columns:
                    universe.loc[selected, "source_type"] = "ifind_index_daily"
                if "note" in universe.columns:
                    universe.loc[selected, "note"] = spec["proxy_note"]
            universe.to_parquet(universe_path, index=False)
        print(f"Applied with backup: {archive.relative_to(ROOT)}", flush=True)
    else:
        print("Dry run only. Re-run with --apply after reviewing the coverage report.", flush=True)


if __name__ == "__main__":
    main()
