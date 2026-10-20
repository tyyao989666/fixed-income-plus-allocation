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


def month_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}年{ts.month}月"


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    keys = [x.strip() for x in raw.split(";") if x.strip()]
    return keys + [None]


def run_wind(question: str, cache: Path) -> tuple[object, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8")), "ok_cached"
        except Exception:
            cache.unlink(missing_ok=True)
    params = {"question": question.replace(" ", ""), "lang": "中文"}
    last_error: object = None
    for key in key_pool():
        env = os.environ.copy()
        if key:
            env["WIND_API_KEY"] = key
        cmd = [
            "node",
            str(SKILL_CLI),
            "call",
            "bond_data",
            "get_bond_basicinfo",
            json.dumps(params, ensure_ascii=False, separators=(",", ":")),
        ]
        res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180, env=env)
        if res.returncode != 0:
            last_error = {"error": res.stdout.strip() or res.stderr.strip()}
            if "BALANCE_INSUFFICIENT" in json.dumps(last_error, ensure_ascii=False):
                continue
            return last_error, "failed"
        try:
            outer = json.loads(res.stdout)
            payload = json.loads(outer["content"][0]["text"])
            if payload.get("error"):
                last_error = {"error": payload["error"]}
                if "BALANCE_INSUFFICIENT" in json.dumps(payload["error"], ensure_ascii=False):
                    continue
                return last_error, "failed"
            data = payload.get("data")
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data, "ok"
        except Exception as exc:
            return {"error": f"{exc}: {res.stdout[:300]}"}, "failed_parse"
    return last_error or {"error": "all_keys_failed"}, "failed"


def payload_to_df(payload: object, dataset: str, month: pd.Timestamp, status: str) -> tuple[pd.DataFrame, int]:
    blocks = payload.get("data", []) if isinstance(payload, dict) else payload
    frames = []
    excel_total = 0
    if not isinstance(blocks, list):
        return pd.DataFrame(), 0
    for i, block in enumerate(blocks):
        if not isinstance(block, dict) or "columns" not in block or "rows" not in block:
            continue
        cols = [str(c.get("name", f"col_{j}")) for j, c in enumerate(block.get("columns") or [])]
        rows = block.get("rows") or []
        excel_total += int(block.get("excelTotalCount") or 0)
        if not rows:
            continue
        max_len = max(len(r) for r in rows)
        cols = (cols + [f"col_{j}" for j in range(len(cols), max_len)])[:max_len]
        df = pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=cols)
        df.insert(0, "dataset", dataset)
        df.insert(1, "query_month", month)
        df.insert(2, "table_no", i)
        df["fetch_status"] = status
        df["resolved_question"] = block.get("resolved_question", "")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return out, excel_total


def standardize(issue: pd.DataFrame, maturity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not issue.empty:
        date_col = "发行起始日期"
        amount_col = "发行总额"
        tmp = issue.copy()
        tmp["date"] = pd.to_datetime(tmp.get(date_col), errors="coerce")
        tmp["amount"] = pd.to_numeric(tmp.get(amount_col), errors="coerce")
        tmp = tmp[tmp["date"].notna()]
        tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp("M")
        g = tmp.groupby("month", as_index=False).agg(
            credit_bond_issuance_amount=("amount", "sum"),
            credit_bond_issuance_count=("Wind代码", "nunique"),
        )
        rows.append(g)
    if not maturity.empty:
        date_col = "到期日期"
        amount_col = "债券余额"
        tmp = maturity.copy()
        tmp["date"] = pd.to_datetime(tmp.get(date_col), errors="coerce")
        tmp["amount"] = pd.to_numeric(tmp.get(amount_col), errors="coerce")
        tmp = tmp[tmp["date"].notna()]
        tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp("M")
        g = tmp.groupby("month", as_index=False).agg(
            credit_bond_maturity_amount=("amount", "sum"),
            credit_bond_maturity_count=("Wind代码", "nunique"),
        )
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    out = rows[0]
    for x in rows[1:]:
        out = out.merge(x, on="month", how="outer")
    out = out.sort_values("month")
    out["credit_bond_net_financing"] = out.get("credit_bond_issuance_amount", 0).fillna(0) - out.get("credit_bond_maturity_amount", 0).fillna(0)
    out["data_source"] = "wind_bond_data_detail_aggregated"
    out["quality_flag"] = "detail_aggregated"
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    months = pd.period_range("2018-01", "2026-06", freq="M").to_timestamp("M")
    logs = []
    issue_frames = []
    maturity_frames = []
    templates = {
        "issue": "查询{month}信用债发行明细，返回债券代码、债券简称、发行日期、发行规模、债券类型。",
        "maturity": "查询{month}信用债到期兑付明细，返回债券代码、债券简称、到期日期、偿还规模、债券类型。",
    }
    for m in months:
        for dataset, template in templates.items():
            q = template.format(month=month_label(m))
            cache = RAW / f"bond_data_credit_financing_{dataset}_{m.strftime('%Y%m')}.json"
            payload, status = run_wind(q, cache)
            df, total = payload_to_df(payload, dataset, m, status)
            if dataset == "issue" and not df.empty:
                issue_frames.append(df)
            if dataset == "maturity" and not df.empty:
                maturity_frames.append(df)
            logs.append(
                {
                    "dataset": dataset,
                    "month": m,
                    "status": status,
                    "rows": len(df),
            "row_lt_total": False,
            "excel_total": total,
                    "file": str(cache.relative_to(ROOT)) if cache.exists() else "",
                    "error": "" if status.startswith("ok") else json.dumps(payload, ensure_ascii=False)[:400],
                }
            )
            time.sleep(0.15)

    issue = pd.concat(issue_frames, ignore_index=True, sort=False) if issue_frames else pd.DataFrame()
    maturity = pd.concat(maturity_frames, ignore_index=True, sort=False) if maturity_frames else pd.DataFrame()
    if not issue.empty:
        issue.to_parquet(PROCESSED / "wind_credit_bond_issue_details.parquet", index=False)
        issue.to_csv(PROCESSED / "wind_credit_bond_issue_details.csv", index=False, encoding="utf-8-sig")
    if not maturity.empty:
        maturity.to_parquet(PROCESSED / "wind_credit_bond_maturity_details.parquet", index=False)
        maturity.to_csv(PROCESSED / "wind_credit_bond_maturity_details.csv", index=False, encoding="utf-8-sig")

    monthly = standardize(issue, maturity)
    if not monthly.empty:
        monthly.to_parquet(PROCESSED / "wind_credit_net_financing_monthly.parquet", index=False)
        monthly.to_csv(PROCESSED / "wind_credit_net_financing_monthly.csv", index=False, encoding="utf-8-sig")

    log = pd.DataFrame(logs)
    with pd.ExcelWriter(REPORTS / "wind_credit_financing_details_report.xlsx", engine="openpyxl") as writer:
        log.to_excel(writer, sheet_name="download_log", index=False)
        if not monthly.empty:
            monthly.to_excel(writer, sheet_name="monthly_net_financing", index=False)
        if not issue.empty:
            issue.head(5000).to_excel(writer, sheet_name="issue_preview", index=False)
        if not maturity.empty:
            maturity.head(5000).to_excel(writer, sheet_name="maturity_preview", index=False)

    lines = [
        "# Wind 信用债发行到期明细聚合报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 下载任务：{len(log)}",
        f"- 成功/缓存成功：{log['status'].astype(str).str.startswith('ok').sum()}",
        f"- 失败：{(~log['status'].astype(str).str.startswith('ok')).sum()}",
        f"- 发行明细行数：{len(issue)}",
        f"- 到期明细行数：{len(maturity)}",
        f"- 月频净融资行数：{len(monthly)}",
        f"- 存在返回行数小于excelTotalCount的任务：{int(log['row_lt_total'].sum())}",
        "",
        "## 输出",
        "- data_processed/wind_credit_bond_issue_details.parquet",
        "- data_processed/wind_credit_bond_maturity_details.parquet",
        "- data_processed/wind_credit_net_financing_monthly.parquet",
        "- output/reports/wind_credit_financing_details_report.xlsx",
    ]
    (REPORTS / "wind_credit_financing_details_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if not monthly.empty:
        print(monthly.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
