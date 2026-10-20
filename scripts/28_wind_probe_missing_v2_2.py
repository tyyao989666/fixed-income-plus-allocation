from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
START_DATE = "20180101"
END_DATE = "20260603"


def run_wind(server: str, tool: str, params: dict, cache: Path) -> tuple[str, int, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            return "ok_cached", count_rows(payload), ",".join(extract_columns(payload))
        except Exception:
            pass
    cmd = ["node", str(SKILL_CLI), "call", server, tool, json.dumps(params, ensure_ascii=False, separators=(",", ":"))]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180)
    if res.returncode != 0:
        return f"failed:{res.stdout.strip() or res.stderr.strip()}", 0, ""
    try:
        outer = json.loads(res.stdout)
        text = outer["content"][0]["text"]
        payload = json.loads(text)
        if payload.get("error"):
            return f"failed:{json.dumps(payload['error'], ensure_ascii=False)}", 0, ""
        data = payload.get("data")
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return "ok", count_rows(data), ",".join(extract_columns(data))
    except Exception as exc:
        return f"failed_parse:{exc}:{res.stdout[:300]}", 0, ""


def count_rows(payload) -> int:
    if payload is None:
        return 0
    if isinstance(payload, dict) and "rows" in payload:
        return len(payload.get("rows", []))
    if isinstance(payload, dict) and "data" in payload:
        return sum(count_rows(x) for x in payload.get("data", []) if isinstance(x, dict))
    if isinstance(payload, list):
        return sum(count_rows(x) for x in payload)
    return 1


def extract_columns(payload) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, dict) and "columns" in payload:
        return [c.get("name", "") for c in payload.get("columns", [])]
    if isinstance(payload, dict) and "data" in payload:
        cols: list[str] = []
        for x in payload.get("data", []):
            cols.extend(extract_columns(x))
        return cols[:80]
    if isinstance(payload, list):
        cols: list[str] = []
        for x in payload:
            cols.extend(extract_columns(x))
        return cols[:80]
    return ["raw_json"]


def main() -> None:
    raw = ROOT / "data_raw" / "wind"
    reports = ROOT / "output" / "reports"
    probes = []

    analytics_questions = {
        "nanhua_commodity_analytics": "查询2018年1月1日至2026年6月3日南华商品指数日频收盘价数据，返回日期、指标名称、指标代码和值。",
        "gold_spot_analytics": "查询2018年1月1日至2026年6月3日上海黄金交易所Au99.99或上海金日频收盘价数据，返回日期、指标名称、指标代码和值。",
        "usd_index_analytics": "查询2018年1月1日至2026年6月3日美元指数日频收盘价数据，返回日期、指标名称、指标代码和值。",
        "us_10y_yield_analytics": "查询2018年1月1日至2026年6月3日美国10年期国债收益率日频数据，返回日期、指标名称、指标代码和值。",
        "us_real_rate_analytics": "查询2018年1月1日至2026年6月3日美国10年期实际利率日频数据，返回日期、指标名称、指标代码和值。",
        "aaa_cd_1y_analytics_retry": "查询2018年1月1日至2026年6月3日中债商业银行同业存单到期收益率(AAA):1年或1年期AAA同业存单收益率数据，返回日期、指标名称、指标代码和值。",
        "cb_premium_median_analytics": "查询2018年1月1日至2026年6月3日可转债市场转股溢价率中位数日频数据，返回日期、指标名称、指标代码和值。",
        "cb_price_median_analytics": "查询2018年1月1日至2026年6月3日可转债市场价格中位数、平均价格、YTM中位数、纯债溢价率中位数日频数据，返回日期、指标名称、指标代码和值。",
        "cb_balance_issuance_call_analytics": "查询2018年1月1日至2026年6月3日可转债市场余额、新券发行规模、强赎退出规模月频数据，返回日期、指标名称、指标代码和值。",
        "credit_financing_analytics": "查询2018年1月1日至2026年6月3日信用债净融资、信用债到期规模、取消发行规模月频数据，返回日期、指标名称、指标代码和值。",
        "rating_downgrade_analytics": "查询2018年1月1日至2026年6月3日信用债评级下调数量、评级负面事件数量月频数据，返回日期、指标名称、指标代码和值。",
        "industrial_output_export_fai_analytics": "查询2018年1月至2026年6月中国工业增加值同比、固定资产投资同比、出口同比月频数据，返回数据期、发布日期、指标名称和值。",
    }
    for name, question in analytics_questions.items():
        cache = raw / f"analytics_probe_{name}_{START_DATE}_{END_DATE}.json"
        status, rows, cols = run_wind("analytics_data", "get_financial_data", {"question": question, "lang": "中文"}, cache)
        probes.append({"probe": name, "server": "analytics_data", "tool": "get_financial_data", "rows": rows, "columns": cols, "file": str(cache.relative_to(ROOT)), "status": status})
        time.sleep(0.15)

    kline_candidates = {
        "nanhua_commodity": ["南华商品指数", "NHCI.NH", "NH0100.NHF", "NH0000.NH"],
        "gold": ["上海金", "Au99.99", "AU9999.SGE", "AU9999.SHF", "黄金9999"],
        "usd_index": ["美元指数", "USDX.FX", "DXY.GI", "DXY"],
        "us_10y": ["美国10年期国债收益率", "USGG10YR.GB", "UST10Y.GB", "US10YT.GB"],
    }
    for group, codes in kline_candidates.items():
        for code in codes:
            cache = raw / f"index_probe_{group}_{code.replace(':','_').replace('/','_')}_{START_DATE}_{END_DATE}.json"
            params = {"windcode": code, "begin_date": START_DATE, "end_date": END_DATE, "period": "10"}
            status, rows, cols = run_wind("index_data", "get_index_kline", params, cache)
            probes.append({"probe": group, "server": "index_data", "tool": "get_index_kline", "code": code, "rows": rows, "columns": cols, "file": str(cache.relative_to(ROOT)), "status": status})
            time.sleep(0.15)
            if status.startswith("ok") and rows > 0:
                break

    df = pd.DataFrame(probes)
    reports.mkdir(parents=True, exist_ok=True)
    df.to_excel(reports / "v2_2_wind_missing_probe_report.xlsx", index=False)
    lines = [
        "# v2.2 Wind 缺失项探测报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 成功/缓存成功：{df['status'].astype(str).str.startswith('ok').sum()}",
        f"- 失败：{(~df['status'].astype(str).str.startswith('ok')).sum()}",
        "",
        "## 明细",
    ]
    for _, r in df.iterrows():
        lines.append(f"- {r.get('probe')} {r.get('code','')}: rows={r.get('rows')} status={r.get('status')} file={r.get('file')}")
    (reports / "v2_2_wind_missing_probe_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(df[["probe", "code", "rows", "status", "file"]].to_string(index=False))


if __name__ == "__main__":
    main()
