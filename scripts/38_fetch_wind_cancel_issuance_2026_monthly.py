from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
RAW = ROOT / "data_raw" / "wind"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


def call(month: int):
    cache = RAW / f"analytics_credit_tail_cancel_issuance_2026_{month:02d}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8")), "ok_cached", cache
    q = f"查询2026年{month}月信用债取消发行总额和取消发行只数，返回取消发行总额、取消发行只数。"
    params = {"question": q, "lang": "中文"}
    cmd = ["node", str(SKILL_CLI), "call", "analytics_data", "get_financial_data", json.dumps(params, ensure_ascii=False, separators=(",", ":"))]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180, env=os.environ.copy())
    if res.returncode != 0:
        return {"error": res.stdout.strip() or res.stderr.strip()}, "failed", cache
    try:
        outer = json.loads(res.stdout)
        payload = json.loads(outer["content"][0]["text"])
        if payload.get("error"):
            return {"error": payload["error"]}, "failed", cache
        data = payload.get("data")
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, "ok", cache
    except Exception as exc:
        return {"error": str(exc)}, "failed_parse", cache


def extract(payload, month: int):
    blocks = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if not isinstance(block, dict) or "columns" not in block or "rows" not in block:
            continue
        cols = [str(c.get("name", "")) for c in block.get("columns") or []]
        rows = block.get("rows") or []
        if not rows:
            continue
        vals = rows[0]
        row = {"month": pd.Timestamp(2026, month, 1) + pd.offsets.MonthEnd(0), "dataset": "cancel_issuance", "data_source": "wind_analytics", "quality_flag": "ok_single_month"}
        for col, val in zip(cols, vals):
            v = pd.to_numeric(val, errors="coerce")
            if pd.isna(v):
                continue
            if "总额" in col or "规模" in col:
                row["cancel_issuance_amount"] = float(v)
            if "只数" in col or "数量" in col:
                row["cancel_issuance_count"] = float(v)
        return row
    return None


def main() -> None:
    rows = []
    log = []
    for m in range(1, 7):
        payload, status, cache = call(m)
        extracted = extract(payload, m) if status.startswith("ok") else None
        if extracted:
            rows.append(extracted)
        log.append({"month": m, "status": status, "file": str(cache.relative_to(ROOT)) if cache.exists() else "", "error": "" if status.startswith("ok") else json.dumps(payload, ensure_ascii=False)[:400]})
        time.sleep(0.2)

    current_path = PROCESSED / "wind_credit_tail_monthly.parquet"
    current = pd.read_parquet(current_path) if current_path.exists() else pd.DataFrame()
    add = pd.DataFrame(rows)
    if not add.empty:
        combined = pd.concat([current, add], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["month", "dataset"], keep="last").sort_values(["dataset", "month"])
        combined.to_parquet(current_path, index=False)
        combined.to_csv(PROCESSED / "wind_credit_tail_monthly.csv", index=False, encoding="utf-8-sig")

    log_df = pd.DataFrame(log)
    log_df.to_excel(REPORTS / "wind_cancel_issuance_2026_monthly_report.xlsx", index=False)
    lines = [
        "# 2026信用债取消发行单月补数报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 成功：{log_df['status'].astype(str).str.startswith('ok').sum()}",
        f"- 失败：{(~log_df['status'].astype(str).str.startswith('ok')).sum()}",
    ]
    (REPORTS / "wind_cancel_issuance_2026_monthly_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    main()
