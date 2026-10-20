from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.data_prep_pipeline import build_monthly_returns


WIND_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
RAW = ROOT / "data_raw" / "wind" / "equity_underlying_indices"

SPECS = [
    {
        "asset_code": "equity_csi1000",
        "asset_name": "中证1000",
        "windcode": "000852.SH",
        "index_name": "中证1000指数",
        "start_date": "2018-01-01",
        "launch_date": "2014-10-17",
        "note": "512100.SH底层指数；指数代理不含ETF跟踪误差与交易成本。",
    },
    {
        "asset_code": "equity_manufacturing",
        "asset_name": "制造业",
        "windcode": "930850.CSI",
        "index_name": "中证智能制造主题指数",
        "start_date": "2018-01-01",
        "launch_date": "2016-07-20",
        "note": "516800.SH底层指数；指数代理不含ETF跟踪误差与交易成本。",
    },
    {
        "asset_code": "equity_tech_growth",
        "asset_name": "科技成长",
        "windcode": "000688.SH",
        "index_name": "上证科创板50成份指数",
        "start_date": "2020-07-23",
        "launch_date": "2020-07-23",
        "note": "588000.SH底层指数；指数发布前不回填，不含ETF跟踪误差与交易成本。",
    },
    {
        "asset_code": "equity_consumption",
        "asset_name": "消费",
        "windcode": "000932.SH",
        "index_name": "中证主要消费指数",
        "start_date": "2018-01-01",
        "launch_date": "2009-07-03",
        "note": "159928.SZ底层指数；指数代理不含ETF跟踪误差与交易成本。",
    },
    {
        "asset_code": "equity_healthcare",
        "asset_name": "医药医疗",
        "windcode": "000933.SH",
        "index_name": "中证医药卫生指数",
        "start_date": "2018-01-01",
        "launch_date": "2009-07-03",
        "note": "512010.SH底层指数；指数代理不含ETF跟踪误差与交易成本。",
    },
]


def call_wind(spec: dict[str, str], end_date: str) -> dict[str, Any]:
    raw_path = RAW / f"{spec['asset_code']}_{spec['windcode']}_{spec['start_date']}_{end_date}.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8"))
    params = {
        "windcode": spec["windcode"],
        "begin_date": spec["start_date"].replace("-", ""),
        "end_date": end_date.replace("-", ""),
        "period": "10",
        "aftype": "0",
        "issusp": "1",
    }
    result = subprocess.run(
        ["node", str(WIND_CLI), "call", "index_data", "get_index_kline", json.dumps(params)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stdout or result.stderr)[-1500:])
    outer = json.loads(result.stdout)
    content = outer.get("content", [])
    if not content:
        raise RuntimeError("Wind returned no content")
    payload = json.loads(content[0]["text"])
    if payload.get("error"):
        raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))
    data = payload.get("data", {})
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def normalize(data: dict[str, Any], spec: dict[str, str], fetch_time: str) -> pd.DataFrame:
    columns = [column["name"] for column in data.get("columns", [])]
    frame = pd.DataFrame(data.get("rows", []), columns=columns)
    if frame.empty:
        raise RuntimeError(f"Empty Wind K-line for {spec['windcode']}")
    date_col = "TIME" if "TIME" in frame.columns else "_DATE"
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    frame["close"] = pd.to_numeric(frame.get("MATCH"), errors="coerce")
    for source, target in (("OPEN", "open"), ("HIGH", "high"), ("LOW", "low"), ("VOLUME", "vol"), ("TURNOVER", "amount")):
        frame[target] = pd.to_numeric(frame.get(source), errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    frame = frame.loc[frame["date"].ge(pd.Timestamp(spec["launch_date"]))].copy()
    frame["return_1d"] = frame["close"].pct_change(fill_method=None)
    return pd.DataFrame(
        {
            "date": frame["date"],
            "asset_code": spec["asset_code"],
            "code": spec["asset_code"],
            "asset_name": spec["asset_name"],
            "asset_class": "equity",
            "module": "equity",
            "sub_module": spec["asset_code"],
            "ts_code": spec["windcode"],
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "vol": frame["vol"],
            "amount": frame["amount"],
            "return_1d": frame["return_1d"],
            "data_source": "wind_underlying_index",
            "fetch_time": fetch_time,
            "member_count": np.nan,
        }
    )


def summarize(frame: pd.DataFrame, version: str) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        selected = frame.loc[frame["asset_code"].astype(str).eq(spec["asset_code"])].copy()
        selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
        ret = pd.to_numeric(selected.get("return_1d"), errors="coerce")
        rows.append(
            {
                "version": version,
                "asset_code": spec["asset_code"],
                "source_code": spec["windcode"],
                "source_index": spec["index_name"],
                "launch_date": spec["launch_date"],
                "rows": len(selected),
                "start": selected["date"].min(),
                "end": selected["date"].max(),
                "annualized_volatility": ret.std() * np.sqrt(252),
                "max_abs_daily_return": ret.abs().max(),
                "note": spec["note"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill ETF histories with point-in-time available underlying indices.")
    parser.add_argument("--end-date", default="2026-07-15")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    fetch_time = datetime.now().isoformat(timespec="seconds")
    daily_path = PROCESSED / "asset_returns_daily.parquet"
    daily = pd.read_parquet(daily_path)
    before = summarize(daily, "before")
    replacement = pd.concat([normalize(call_wind(spec, args.end_date), spec, fetch_time) for spec in SPECS], ignore_index=True)
    target_codes = {spec["asset_code"] for spec in SPECS}
    output = pd.concat(
        [daily.loc[~daily["asset_code"].astype(str).isin(target_codes)], replacement],
        ignore_index=True,
        sort=False,
    )
    for column in daily.columns:
        if column not in output.columns:
            output[column] = np.nan
    output = output[daily.columns].sort_values(["date", "asset_code"]).reset_index(drop=True)
    pd.concat([before, summarize(output, "after")], ignore_index=True).to_csv(
        REPORTS / "equity_underlying_index_backfill_coverage.csv", index=False, encoding="utf-8-sig"
    )
    for spec in SPECS:
        selected = replacement.loc[replacement["asset_code"].eq(spec["asset_code"])]
        print(f"{spec['asset_code']}: {len(selected)} rows {selected['date'].min().date()}..{selected['date'].max().date()}")
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the coverage report.")
        return
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = PROCESSED / "archive" / f"pre_equity_index_backfill_{tag}"
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(daily_path, archive / daily_path.name)
    monthly_path = PROCESSED / "asset_returns_monthly.parquet"
    universe_path = PROCESSED / "asset_universe.parquet"
    if monthly_path.exists():
        shutil.copy2(monthly_path, archive / monthly_path.name)
    if universe_path.exists():
        shutil.copy2(universe_path, archive / universe_path.name)
    output.to_parquet(daily_path, index=False)
    build_monthly_returns(output).to_parquet(monthly_path, index=False)
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        code_column = "asset_code" if "asset_code" in universe.columns else "code"
        for spec in SPECS:
            selected = universe[code_column].astype(str).eq(spec["asset_code"])
            universe.loc[selected, "ts_code"] = spec["windcode"]
            universe.loc[selected, "data_source"] = "wind_underlying_index"
            if "source_type" in universe.columns:
                universe.loc[selected, "source_type"] = "wind_index_daily"
            if "note" in universe.columns:
                universe.loc[selected, "note"] = spec["note"]
        universe.to_parquet(universe_path, index=False)
    print(f"Applied with backup: {archive.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
