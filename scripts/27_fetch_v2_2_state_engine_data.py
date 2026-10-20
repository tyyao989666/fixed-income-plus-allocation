from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
START_DATE = "20180101"
END_DATE = "20260603"
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")


def pro_api() -> Any:
    token = ts.get_token()
    if not token:
        raise RuntimeError("Tushare token missing")
    return ts.pro_api(token)


def save_df(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()
    df.to_parquet(path, index=False)
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": ",".join(df.columns.astype(str)) if not df.empty else "",
    }


def fetch_tushare() -> list[dict[str, Any]]:
    pro = pro_api()
    out: list[dict[str, Any]] = []
    raw = ROOT / "data_raw" / "tushare"

    calls = [
        ("trade_cal", "calendar", {"exchange": "SSE", "start_date": START_DATE, "end_date": END_DATE}),
        ("shibor", "shibor", {"start_date": START_DATE, "end_date": END_DATE}),
        ("cn_m", "money_supply", {"start_m": START_DATE[:6], "end_m": END_DATE[:6]}),
        ("cn_cpi", "cpi", {"start_m": START_DATE[:6], "end_m": END_DATE[:6]}),
        ("cn_ppi", "ppi", {"start_m": START_DATE[:6], "end_m": END_DATE[:6]}),
        ("cn_pmi", "pmi", {"start_m": START_DATE[:6], "end_m": END_DATE[:6]}),
        ("sf_month", "social_financing", {"start_m": START_DATE[:6], "end_m": END_DATE[:6]}),
        ("margin", "margin_balance", {"start_date": START_DATE, "end_date": END_DATE}),
    ]
    for endpoint, key, params in calls:
        path = raw / f"{endpoint}_{key}_{START_DATE}_{END_DATE}.parquet"
        try:
            df = pd.read_parquet(path) if path.exists() else getattr(pro, endpoint)(**params)
            meta = save_df(df, path)
            meta.update({"source": "tushare", "endpoint": endpoint, "status": "ok"})
        except Exception as exc:
            meta = {"source": "tushare", "endpoint": endpoint, "file": str(path.relative_to(ROOT)), "rows": 0, "columns": "", "status": f"failed:{exc}"}
        out.append(meta)
        time.sleep(0.08)

    index_codes = {
        "hs300": "000300.SH",
        "csi500": "000905.SH",
        "csi1000": "000852.SH",
        "chinext": "399006.SZ",
        "sse": "000001.SH",
        "dividend": "000922.CSI",
        "csi_cb": "000832.CSI",
    }
    for name, code in index_codes.items():
        for endpoint in ["index_daily", "index_dailybasic"]:
            path = raw / f"{endpoint}_{name}_{START_DATE}_{END_DATE}.parquet"
            try:
                df = pd.read_parquet(path) if path.exists() else getattr(pro, endpoint)(ts_code=code, start_date=START_DATE, end_date=END_DATE)
                meta = save_df(df, path)
                meta.update({"source": "tushare", "endpoint": endpoint, "status": "ok", "code": code})
            except Exception as exc:
                meta = {"source": "tushare", "endpoint": endpoint, "file": str(path.relative_to(ROOT)), "rows": 0, "columns": "", "status": f"failed:{exc}", "code": code}
            out.append(meta)
            time.sleep(0.08)

    # Optional macro endpoints vary by token/version; keep failures explicit.
    optional_calls = [
        ("cn_gdp", "gdp", {"start_q": "2018Q1", "end_q": "2026Q2"}),
        ("cn_industry", "industrial_production", {"start_m": START_DATE[:6], "end_m": END_DATE[:6]}),
        ("cn_ppi", "ppi_repeat_for_release_check", {"start_m": START_DATE[:6], "end_m": END_DATE[:6]}),
        ("fx_daily", "usdcny", {"ts_code": "USDCNY.FXCM", "start_date": START_DATE, "end_date": END_DATE}),
    ]
    for endpoint, key, params in optional_calls:
        path = raw / f"{endpoint}_{key}_{START_DATE}_{END_DATE}.parquet"
        try:
            df = pd.read_parquet(path) if path.exists() else getattr(pro, endpoint)(**params)
            meta = save_df(df, path)
            meta.update({"source": "tushare", "endpoint": endpoint, "status": "ok"})
        except Exception as exc:
            meta = {"source": "tushare", "endpoint": endpoint, "file": str(path.relative_to(ROOT)), "rows": 0, "columns": "", "status": f"failed:{exc}"}
        out.append(meta)
        time.sleep(0.08)

    return out


def call_wind(server: str, tool: str, params: dict[str, Any], cache_path: Path) -> tuple[dict[str, Any] | None, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), "ok_cached"
    cmd = ["node", str(SKILL_CLI), "call", server, tool, json.dumps(params, ensure_ascii=False, separators=(",", ":"))]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180)
    if res.returncode != 0:
        return None, f"failed:{res.stdout.strip() or res.stderr.strip()}"
    try:
        outer = json.loads(res.stdout)
        text = outer["content"][0]["text"]
        payload = json.loads(text)
        if payload.get("error"):
            return None, f"failed:{json.dumps(payload['error'], ensure_ascii=False)}"
        data = payload.get("data")
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, "ok"
    except Exception as exc:
        return None, f"failed_parse:{exc}:{res.stdout[:500]}"


def wind_rows_to_df(data: dict[str, Any]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    cols = [c.get("name") for c in data.get("columns", [])]
    return pd.DataFrame(data.get("rows", []), columns=cols)


def fetch_wind() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    raw = ROOT / "data_raw" / "wind"
    tables = ROOT / "output" / "tables"

    index_targets = {
        "wind_all_a": "881001.WI",
        "csi_convertible_bond": "000832.CSI",
        "dividend_index": "000922.CSI",
        "nanhua_commodity": "NHCI.NH",
        "gold_spot_proxy": "AU9999.SGE",
        "vix": "VIX.GI",
        "dollar_index": "USDX.FX",
        "us_10y_yield_index_proxy": "USGG10YR.GB",
    }
    for name, windcode in index_targets.items():
        cache = raw / f"index_kline_{name}_{windcode}_{START_DATE}_{END_DATE}.json"
        params = {"windcode": windcode, "begin_date": START_DATE, "end_date": END_DATE, "period": "10"}
        data, status = call_wind("index_data", "get_index_kline", params, cache)
        df = wind_rows_to_df(data) if data else pd.DataFrame()
        table_path = tables / f"wind_{name}.parquet"
        if not df.empty:
            save_df(df, table_path)
        out.append({"source": "wind", "endpoint": "index_kline", "code": windcode, "file": str(cache.relative_to(ROOT)), "derived_table": str(table_path.relative_to(ROOT)), "rows": int(len(df)), "columns": ",".join(df.columns.astype(str)) if not df.empty else "", "status": status})
        time.sleep(0.12)

    nl_queries = {
        "treasury_yield_curve": "查询2018年1月1日至2026年6月3日中国1年、3年、5年、10年、30年国债收益率日频数据，返回日期、指标名称、指标代码和值。",
        "dr007_r007": "查询2018年1月1日至2026年6月3日DR007、R007、R007-DR007日频数据，返回日期、指标名称、指标代码和值。",
        "aaa_cd_1y": "查询2018年1月1日至2026年6月3日1年AAA同业存单收益率日频数据，返回日期、指标名称、指标代码和值。",
        "credit_spreads": "查询2018年1月1日至2026年6月3日AAA、AA+、AA信用利差、城投债利差、产业债利差日频或周频数据，返回日期、指标名称、指标代码和值。",
        "cb_valuation": "查询2018年1月1日至2026年6月3日可转债市场转股溢价率中位数、纯债溢价率中位数、YTM中位数、市场余额、新券发行规模、强赎退出规模数据，返回日期、指标名称、指标代码和值。",
        "omo_policy": "查询2018年1月1日至2026年6月3日公开市场操作净投放、MLF利率、7天逆回购利率数据，返回日期、指标名称、指标代码和值。",
        "macro_release": "查询2018年至2026年中国工业增加值同比、固定资产投资同比、出口同比，以及对应发布日期或披露日期，返回数据期、发布日期和值。",
    }
    for name, question in nl_queries.items():
        cache = raw / f"analytics_{name}_{START_DATE}_{END_DATE}.json"
        data, status = call_wind("analytics_data", "get_financial_data", {"question": question, "lang": "中文"}, cache)
        rows = 0
        cols = ""
        if data is not None:
            if isinstance(data, dict) and "rows" in data:
                rows = len(data.get("rows", []))
                cols = ",".join([c.get("name", "") for c in data.get("columns", [])])
            else:
                rows = 1
                cols = "raw_json"
        out.append({"source": "wind", "endpoint": "analytics_data", "code": name, "file": str(cache.relative_to(ROOT)), "rows": rows, "columns": cols, "status": status})
        time.sleep(0.12)
    return out


def build_coverage(meta: list[dict[str, Any]]) -> None:
    processed = ROOT / "data_processed"
    reports = ROOT / "output" / "reports"
    processed.mkdir(exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(meta)
    df["fetch_time"] = datetime.now().isoformat(timespec="seconds")
    df.to_parquet(processed / "state_engine_raw_coverage.parquet", index=False)
    with pd.ExcelWriter(reports / "v2_2_state_engine_download_report.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="download_status", index=False)
        summary = df.groupby(["source", "endpoint", "status"], dropna=False).size().reset_index(name="count")
        summary.to_excel(writer, sheet_name="summary", index=False)
    ok = df[df["status"].astype(str).str.startswith("ok")]
    fail = df[~df["status"].astype(str).str.startswith("ok")]
    lines = [
        "# v2.2 状态引擎补数报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 成功/缓存成功：{len(ok)} 项",
        f"- 失败/待人工确认：{len(fail)} 项",
        "",
        "## 成功项",
    ]
    for _, r in ok.iterrows():
        lines.append(f"- {r.get('source')} {r.get('endpoint')} {r.get('code','')}: rows={r.get('rows')}, file={r.get('file')}")
    lines.append("")
    lines.append("## 失败项")
    if fail.empty:
        lines.append("- 无")
    else:
        for _, r in fail.iterrows():
            lines.append(f"- {r.get('source')} {r.get('endpoint')} {r.get('code','')}: {r.get('status')}")
    (reports / "v2_2_state_engine_download_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    meta = []
    meta.extend(fetch_tushare())
    meta.extend(fetch_wind())
    build_coverage(meta)
    print(pd.DataFrame(meta)[["source", "endpoint", "code", "rows", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
