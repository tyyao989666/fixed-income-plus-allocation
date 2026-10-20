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


def run_wind(question: str, cache: Path) -> tuple[str, int, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return "ok_cached", count_rows(data), ",".join(extract_columns(data))
        except Exception:
            cache.unlink(missing_ok=True)

    params = {"question": question.replace(" ", ""), "lang": "中文"}
    cmd = [
        "node",
        str(SKILL_CLI),
        "call",
        "analytics_data",
        "get_financial_data",
        json.dumps(params, ensure_ascii=False, separators=(",", ":")),
    ]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180)
    if res.returncode != 0:
        msg = (res.stdout.strip() or res.stderr.strip()).replace("\n", " ")[:500]
        return f"failed:{msg}", 0, ""
    try:
        outer = json.loads(res.stdout)
        text = outer["content"][0]["text"]
        payload = json.loads(text)
        if payload.get("error"):
            return f"failed:{json.dumps(payload['error'], ensure_ascii=False)[:500]}", 0, ""
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
        return [str(c.get("name", "")) for c in payload.get("columns", [])]
    if isinstance(payload, dict) and "data" in payload:
        cols: list[str] = []
        for x in payload.get("data", []):
            cols.extend(extract_columns(x))
        return cols[:120]
    if isinstance(payload, list):
        cols: list[str] = []
        for x in payload:
            cols.extend(extract_columns(x))
        return cols[:120]
    return ["raw_json"]


def main() -> None:
    raw = ROOT / "data_raw" / "wind"
    reports = ROOT / "output" / "reports"
    questions = {
        "cb_balance_stock_monthly": "查询2018年1月至2026年6月可转债市场存量余额月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "cb_balance_stock_daily": "查询2018年1月1日至2026年6月3日可转债市场存量余额日频数据,返回日期、指标名称、指标代码、数值、单位。",
        "cb_outstanding_amount": "查询2018年1月至2026年6月可转债未偿余额或可转债余额月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "cb_listed_count_balance": "查询2018年1月至2026年6月可转债上市只数、存量只数、市场余额月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "cb_force_redeem_count": "查询2018年1月至2026年6月可转债强制赎回只数、强赎数量月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "cb_force_redeem_amount": "查询2018年1月至2026年6月可转债强制赎回规模、强赎退市规模月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "cb_delist_redeem": "查询2018年1月至2026年6月可转债退市数量、赎回退市数量、到期退市数量月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "credit_net_financing_syn1": "查询2018年1月至2026年6月信用债净融资额月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "credit_net_financing_syn2": "查询2018年1月至2026年6月信用债发行量、偿还量、净融资月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "corp_bond_net_financing": "查询2018年1月至2026年6月企业债、公司债、中期票据、短期融资券净融资额月频数据,返回日期、品种、发行量、偿还量、净融资额。",
        "bond_cancel_amount": "查询2018年1月至2026年6月债券取消发行规模月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "bond_cancel_count": "查询2018年1月至2026年6月债券取消发行只数月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "credit_rating_downgrade_count": "查询2018年1月至2026年6月信用债主体评级下调数量月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "issuer_rating_downgrade_event": "查询2018年1月至2026年6月债券发行人主体评级下调事件数量月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "bond_default_amount": "查询2018年1月至2026年6月信用债违约规模、违约只数月频数据,返回日期、指标名称、指标代码、数值、单位。",
        "cb_balance_recent": "查询2025年1月至2026年6月可转债市场余额、存量只数、新券发行、退市、强赎月频数据。",
        "credit_tail_recent": "查询2025年1月至2026年6月信用债净融资、到期规模、取消发行规模、评级下调数量、违约规模月频数据。",
    }

    rows = []
    for name, question in questions.items():
        cache = raw / f"analytics_deep_probe_{name}_{START_DATE}_{END_DATE}.json"
        status, row_count, columns = run_wind(question, cache)
        rows.append(
            {
                "probe": name,
                "rows": row_count,
                "columns": columns,
                "file": str(cache.relative_to(ROOT)) if cache.exists() else "",
                "status": status,
            }
        )
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    reports.mkdir(parents=True, exist_ok=True)
    xlsx = reports / "v2_2_wind_deep_probe_remaining_report.xlsx"
    md = reports / "v2_2_wind_deep_probe_remaining_report.md"
    df.to_excel(xlsx, index=False)
    lines = [
        "# v2.2 Wind 剩余缺口深度探测报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 成功/缓存成功：{df['status'].astype(str).str.startswith('ok').sum()}",
        f"- 失败：{(~df['status'].astype(str).str.startswith('ok')).sum()}",
        "",
        "## 明细",
    ]
    for _, r in df.iterrows():
        lines.append(f"- {r['probe']}: rows={r['rows']} status={r['status']} file={r['file']}")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(df[["probe", "rows", "status", "file"]].to_string(index=False))
    print(f"\nSaved: {xlsx}")
    print(f"Saved: {md}")


if __name__ == "__main__":
    main()
