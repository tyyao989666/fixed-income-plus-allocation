from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
RAW = ROOT / "data_raw" / "wind"
REPORTS = ROOT / "output" / "reports"


def call(question: str, name: str) -> dict:
    cache = RAW / f"analytics_credit_tail_month_probe_{name}.json"
    if cache.exists():
        return {"name": name, "status": "ok_cached", "file": str(cache.relative_to(ROOT)), "payload": json.loads(cache.read_text(encoding="utf-8"))}
    params = {"question": question.replace(" ", ""), "lang": "中文"}
    cmd = ["node", str(SKILL_CLI), "call", "analytics_data", "get_financial_data", json.dumps(params, ensure_ascii=False, separators=(",", ":"))]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180, env=os.environ.copy())
    if res.returncode != 0:
        return {"name": name, "status": "failed", "error": res.stdout.strip() or res.stderr.strip()}
    try:
        outer = json.loads(res.stdout)
        payload = json.loads(outer["content"][0]["text"])
        if payload.get("error"):
            return {"name": name, "status": "failed", "error": json.dumps(payload["error"], ensure_ascii=False)}
        data = payload.get("data")
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"name": name, "status": "ok", "file": str(cache.relative_to(ROOT)), "payload": data}
    except Exception as exc:
        return {"name": name, "status": "failed_parse", "error": f"{exc}:{res.stdout[:300]}"}


def count_rows(payload) -> int:
    if isinstance(payload, dict) and "rows" in payload:
        return len(payload.get("rows") or [])
    if isinstance(payload, dict) and "data" in payload:
        return sum(count_rows(x) for x in payload.get("data") or [])
    if isinstance(payload, list):
        return sum(count_rows(x) for x in payload)
    return 0


def cols(payload) -> str:
    if isinstance(payload, dict) and "columns" in payload:
        return ",".join(str(c.get("name", "")) for c in payload.get("columns") or [])
    if isinstance(payload, dict) and "data" in payload:
        return " | ".join(filter(None, [cols(x) for x in payload.get("data") or []]))
    if isinstance(payload, list):
        return " | ".join(filter(None, [cols(x) for x in payload]))
    return ""


def main() -> None:
    tests = {
        "net_202501": "查询2025年1月信用债发行规模、偿还规模、净融资额，返回月份、发行规模、偿还规模、净融资额。",
        "net_202506": "查询2025年6月信用债发行规模、偿还规模、净融资额，返回月份、发行规模、偿还规模、净融资额。",
        "cancel_202601": "查询2026年1月信用债取消发行规模、取消发行只数，返回月份、取消发行规模、取消发行只数。",
        "cancel_202605": "查询2026年5月信用债取消发行规模、取消发行只数，返回月份、取消发行规模、取消发行只数。",
        "rating_202501": "查询2025年1月信用债主体评级下调数量，返回月份、评级下调数量。",
        "rating_202506": "查询2025年6月信用债主体评级下调数量，返回月份、评级下调数量。",
        "rating_neg_202501": "查询2025年1月债券发行人负面评级行动数量，返回月份、负面评级行动数量。",
    }
    rows = []
    for name, q in tests.items():
        r = call(q, name)
        payload = r.pop("payload", None)
        rows.append({**r, "rows": count_rows(payload), "columns": cols(payload), "error": r.get("error", "")})
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_excel(REPORTS / "wind_credit_tail_month_probe_report.xlsx", index=False)
    df.to_csv(REPORTS / "wind_credit_tail_month_probe_report.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
