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


def run_wind(question: str, cache: Path) -> tuple[object, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8")), "ok_cached"
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
    res = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=180,
        env=os.environ.copy(),
    )
    if res.returncode != 0:
        return {"error": res.stdout.strip() or res.stderr.strip()}, "failed"
    try:
        outer = json.loads(res.stdout)
        payload = json.loads(outer["content"][0]["text"])
        if payload.get("error"):
            return {"error": payload["error"]}, "failed"
        data = payload.get("data")
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, "ok"
    except Exception as exc:
        return {"error": f"{exc}: {res.stdout[:300]}"}, "failed_parse"


def table_frames(payload: object, dataset: str, year: int, status: str) -> list[pd.DataFrame]:
    frames = []
    blocks = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(blocks, list):
        return frames
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict) or "columns" not in block or "rows" not in block:
            continue
        rows = block.get("rows") or []
        if not rows:
            continue
        cols = [str(c.get("name", f"col_{i}")) for i, c in enumerate(block.get("columns") or [])]
        max_len = max(len(r) for r in rows)
        cols = (cols + [f"col_{i}" for i in range(len(cols), max_len)])[:max_len]
        df = pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=cols)
        df.insert(0, "dataset", dataset)
        df.insert(1, "source_year", year)
        df.insert(2, "table_no", idx)
        df["resolved_question"] = block.get("resolved_question", "")
        df["fetch_status"] = status
        frames.append(df)
    return frames


def standardize_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    out = []
    for _, r in raw.iterrows():
        dataset = r.get("dataset")
        year = pd.to_numeric(r.get("年份", r.get("source_year")), errors="coerce")
        month = pd.to_numeric(r.get("月份", None), errors="coerce")
        if pd.isna(month):
            # Some Wind tables use a period/date column instead of 年份/月份.
            for c in raw.columns:
                if "日期" in str(c) or "月份" in str(c) or "时间" in str(c):
                    dt = pd.to_datetime(r.get(c), errors="coerce")
                    if pd.notna(dt):
                        year, month = dt.year, dt.month
                        break
        if pd.isna(year) or pd.isna(month):
            continue
        row = {
            "month": pd.Timestamp(int(year), int(month), 1) + pd.offsets.MonthEnd(0),
            "data_source": "wind_analytics",
            "dataset": dataset,
        }
        for c in raw.columns:
            name = str(c)
            val = pd.to_numeric(r.get(c), errors="coerce")
            if pd.isna(val):
                continue
            if dataset == "credit_net_financing":
                if "发行" in name and "净" not in name:
                    row["credit_bond_issuance_amount"] = val
                elif "偿还" in name or "到期" in name:
                    row["credit_bond_repayment_amount"] = val
                elif "净融资" in name:
                    row["credit_bond_net_financing"] = val
            elif dataset == "cancel_issuance":
                if "规模" in name or "金额" in name:
                    row["cancel_issuance_amount"] = val
                elif "只数" in name or "数量" in name:
                    row["cancel_issuance_count"] = val
            elif dataset == "rating_downgrade":
                if "下调" in name:
                    row["rating_downgrade_count"] = val
                elif "负面" in name:
                    row["negative_rating_action_count"] = val
            elif dataset == "credit_default":
                if "余额" in name or "规模" in name or "金额" in name:
                    row["credit_default_amount"] = val
                elif "只数" in name or "数量" in name:
                    row["credit_default_count"] = val
        out.append(row)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    numeric_cols = [c for c in df.columns if c not in ["month", "data_source", "dataset"]]
    return df.groupby(["month", "data_source", "dataset"], as_index=False)[numeric_cols].sum(min_count=1)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    templates = {
        "credit_net_financing": "{year}年1月至{year}年{end_month}月信用债月度发行规模、到期偿还规模、净融资额。必须按月份列出，不要汇总。返回年份、月份、发行规模、偿还规模、净融资额。",
        "cancel_issuance": "{year}年1月至{year}年{end_month}月信用债取消发行月度数据。必须按月份列出，不要汇总。返回年份、月份、取消发行规模、取消发行只数。",
        "rating_downgrade": "{year}年1月至{year}年{end_month}月信用债主体评级下调和负面评级行动月度数据。必须按月份列出，不要汇总。返回年份、月份、评级下调数量、负面评级行动数量。",
        "credit_default": "{year}年1月至{year}年{end_month}月信用债违约规模和违约只数月度数据。必须按月份列出，不要汇总。返回年份、月份、违约规模、违约只数。",
    }
    all_frames = []
    report_rows = []
    for dataset, template in templates.items():
        for year in range(2018, 2027):
            end_month = 6 if year == 2026 else 12
            question = template.format(year=year, end_month=end_month)
            cache = RAW / f"analytics_credit_tail_{dataset}_{year}.json"
            payload, status = run_wind(question, cache)
            frames = table_frames(payload, dataset, year, status)
            row_count = sum(len(f) for f in frames)
            report_rows.append(
                {
                    "dataset": dataset,
                    "year": year,
                    "status": status,
                    "rows": row_count,
                    "file": str(cache.relative_to(ROOT)) if cache.exists() else "",
                    "error": "" if status.startswith("ok") else json.dumps(payload, ensure_ascii=False)[:500],
                }
            )
            all_frames.extend(frames)
            time.sleep(0.15)

    raw_df = pd.concat(all_frames, ignore_index=True, sort=False) if all_frames else pd.DataFrame()
    if not raw_df.empty:
        raw_df.to_parquet(PROCESSED / "wind_credit_tail_raw_tables.parquet", index=False)
        raw_df.to_csv(PROCESSED / "wind_credit_tail_raw_tables.csv", index=False, encoding="utf-8-sig")
        monthly = standardize_monthly(raw_df)
    else:
        monthly = pd.DataFrame()
    if not monthly.empty:
        monthly.to_parquet(PROCESSED / "wind_credit_tail_monthly.parquet", index=False)
        monthly.to_csv(PROCESSED / "wind_credit_tail_monthly.csv", index=False, encoding="utf-8-sig")

    report = pd.DataFrame(report_rows)
    with pd.ExcelWriter(REPORTS / "wind_credit_tail_download_report.xlsx", engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="download_log", index=False)
        if not raw_df.empty:
            raw_df.head(5000).to_excel(writer, sheet_name="raw_preview", index=False)
        if not monthly.empty:
            monthly.to_excel(writer, sheet_name="standard_monthly", index=False)

    lines = [
        "# Wind 信用尾部基础数据下载报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 下载任务：{len(report)}",
        f"- 成功/缓存成功：{report['status'].astype(str).str.startswith('ok').sum()}",
        f"- 失败：{(~report['status'].astype(str).str.startswith('ok')).sum()}",
        f"- 原始表行数：{len(raw_df)}",
        f"- 标准月频行数：{len(monthly)}",
        "",
        "## 输出",
        "- data_processed/wind_credit_tail_raw_tables.parquet",
        "- data_processed/wind_credit_tail_monthly.parquet",
        "- output/reports/wind_credit_tail_download_report.xlsx",
        "",
        "## 按数据集统计",
    ]
    if not report.empty:
        for dataset, g in report.groupby("dataset"):
            lines.append(f"- {dataset}: 成功{g['status'].astype(str).str.startswith('ok').sum()}个年份，原始行{int(g['rows'].sum())}")
    (REPORTS / "wind_credit_tail_download_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if not monthly.empty:
        print(monthly.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
