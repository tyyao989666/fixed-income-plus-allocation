from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
RAW_DIR = ROOT / "data_raw" / "wind" / "nanhua_alt_indices"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"

INDICES: dict[str, dict[str, str]] = {
    "nanhua_commodity": {"wind_code": "NH0100.NHF", "name": "南华商品指数"},
    "nanhua_industrial": {"wind_code": "NH0200.NHF", "name": "南华工业品指数"},
    "nanhua_metal": {"wind_code": "NH0400.NHF", "name": "南华金属指数"},
    "nanhua_chemical": {"wind_code": "NH0500.NHF", "name": "南华能化指数"},
    "nanhua_precious_metal": {"wind_code": "NH0600.NHF", "name": "南华贵金属指数"},
    "nanhua_nonferrous": {"wind_code": "NH0700.NHF", "name": "南华有色金属指数"},
}


def project_dates() -> tuple[str, str]:
    cfg_path = ROOT / "config" / "asset_universe.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    project = cfg.get("project", {})
    return str(project.get("start_date", "20180101")), str(project.get("end_date", "20260602"))


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    return [x.strip() for x in raw.split(";") if x.strip()] + [None]


def call_wind_index_kline(wind_code: str, start: str, end: str, cache: Path) -> tuple[dict[str, Any] | list[Any], str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return payload.get("data", []), "ok_cached"

    params = {
        "windcode": wind_code,
        "begin_date": start,
        "end_date": end,
        "period": "10",
    }
    last_error: str | None = None
    for key in key_pool():
        env = os.environ.copy()
        if key:
            env["WIND_API_KEY"] = key
        res = subprocess.run(
            ["node", str(SKILL_CLI), "call", "index_data", "get_index_kline", json.dumps(params, ensure_ascii=False)],
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
            return [], f"failed:{last_error}"
        data = payload.get("data") or []
        cache.write_text(json.dumps({"params": params, "data": data}, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, "ok"
    return [], f"failed:{last_error or 'unknown_error'}"


def normalize_rows(alias: str, meta: dict[str, str], payload: dict[str, Any] | list[Any]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    if isinstance(payload, dict) and "columns" in payload and "rows" in payload:
        cols = [str(c.get("name", f"col_{i}")) for i, c in enumerate(payload.get("columns") or [])]
        rows = payload.get("rows") or []
        df = pd.DataFrame(rows, columns=cols[: len(rows[0])] if rows else cols)
    else:
        df = pd.DataFrame(payload)
    date_col = "_DATE" if "_DATE" in df.columns else "TIME"
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "alias": alias,
            "wind_code": meta["wind_code"],
            "index_name": meta["name"],
            "open": pd.to_numeric(df.get("OPEN"), errors="coerce"),
            "close": pd.to_numeric(df.get("MATCH"), errors="coerce"),
            "high": pd.to_numeric(df.get("HIGH"), errors="coerce"),
            "low": pd.to_numeric(df.get("LOW"), errors="coerce"),
            "volume": pd.to_numeric(df.get("VOLUME"), errors="coerce"),
            "turnover": pd.to_numeric(df.get("TURNOVER"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    # Wind kline sometimes carries exchange-holiday flat rows. Keep real trading rows for return construction.
    trading = out["volume"].fillna(0).gt(0) | out["turnover"].fillna(0).gt(0)
    if trading.any():
        out = out.loc[trading].copy()
    out["return_1d"] = out.groupby("alias")["close"].pct_change()
    out["data_source"] = "wind_index_data.get_index_kline"
    out["fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out.reset_index(drop=True)


def daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for alias, g in daily.sort_values("date").groupby("alias"):
        month_last = g.set_index("date").resample("ME").last().dropna(subset=["close"]).reset_index()
        month_last["return_1m"] = month_last["close"].pct_change()
        rows.append(month_last)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wind Nanhua commodity-family indices.")
    default_start, default_end = project_dates()
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []
    for alias, meta in INDICES.items():
        cache = RAW_DIR / f"{alias}_{meta['wind_code']}_{args.start}_{args.end}.json"
        rows, status = call_wind_index_kline(meta["wind_code"], args.start, args.end, cache)
        df = normalize_rows(alias, meta, rows)
        if not df.empty:
            df.to_parquet(PROCESSED / f"raw_wind_{alias}_daily.parquet", index=False)
            frames.append(df)
        report_rows.append(
            {
                "alias": alias,
                "wind_code": meta["wind_code"],
                "index_name": meta["name"],
                "status": status,
                "rows": int(len(df)),
                "start": df["date"].min().date().isoformat() if not df.empty else "",
                "end": df["date"].max().date().isoformat() if not df.empty else "",
            }
        )

    daily = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    monthly = daily_to_monthly(daily)
    daily.to_parquet(PROCESSED / "wind_nanhua_alt_indices_daily.parquet", index=False)
    monthly.to_parquet(PROCESSED / "wind_nanhua_alt_indices_monthly.parquet", index=False)
    report = pd.DataFrame(report_rows)
    report.to_csv(REPORTS / "wind_nanhua_alt_indices_fetch_report.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"daily_rows": len(daily), "monthly_rows": len(monthly), "report": str((REPORTS / "wind_nanhua_alt_indices_fetch_report.csv").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
