from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


def main() -> None:
    monthly_path = PROCESSED / "wind_credit_tail_monthly.parquet"
    log_path = REPORTS / "wind_credit_tail_download_report.xlsx"
    monthly = pd.read_parquet(monthly_path) if monthly_path.exists() else pd.DataFrame()
    log = pd.read_excel(log_path, sheet_name="download_log") if log_path.exists() else pd.DataFrame()

    usable = []
    gaps = []

    if not monthly.empty and "dataset" in monthly.columns:
        for dataset, g in monthly.groupby("dataset"):
            cols = [c for c in g.columns if c not in ["month", "data_source", "dataset"] and g[c].notna().any()]
            usable.append(
                {
                    "dataset": dataset,
                    "rows": len(g),
                    "start": g["month"].min(),
                    "end": g["month"].max(),
                    "usable_columns": ", ".join(cols),
                    "quality_note": "需要检查单位/口径" if dataset == "credit_net_financing" else "可作为基础状态变量",
                }
            )

    expected = {
        "credit_net_financing": "信用债发行、偿还、净融资月频",
        "cancel_issuance": "信用债取消发行规模/只数月频",
        "rating_downgrade": "主体评级下调/负面评级行动数量月频",
        "credit_default": "信用债违约规模/只数月频",
    }
    for dataset, desc in expected.items():
        l = log[log["dataset"] == dataset] if not log.empty else pd.DataFrame()
        m = monthly[monthly["dataset"] == dataset] if not monthly.empty and "dataset" in monthly.columns else pd.DataFrame()
        status = "downloaded" if not m.empty else "missing"
        issue = ""
        if dataset == "credit_net_financing":
            status = "downloaded_but_not_monthly_reliable" if not m.empty else "missing"
            issue = "Wind自然语言入口每年只返回1行，且偿还/净融资量级疑似年度汇总，不适合直接作月频主口径。"
        elif dataset == "rating_downgrade":
            issue = "Wind返回提取数量超限/没找到数据；iFinD当前授权不允许EDB搜索。"
        elif dataset == "cancel_issuance":
            if not m.empty and pd.to_datetime(m["month"], errors="coerce").max() >= pd.Timestamp("2026-06-30"):
                issue = "2018-01至2026-06月频已补齐。"
            elif not l.empty and (l["status"] == "failed").any():
                issue = "2018-2025已取到月频；2026部分月份需单月补充。"
        gaps.append(
            {
                "dataset": dataset,
                "description": desc,
                "status": status,
                "download_success_years": int(l["status"].astype(str).str.startswith("ok").sum()) if not l.empty else 0,
                "rows_standardized": len(m),
                "issue": issue,
                "recommended_action": {
                    "credit_net_financing": "需要Wind终端EDB指标代码或更高权限；临时可用信用债发行额/偿还量年度口径禁入主模型。",
                    "cancel_issuance": "已可用，作为信用一级市场压力基础变量。",
                    "rating_downgrade": "需要评级机构/Wind专题指标代码；当前先用违约规模、取消发行、利差压力做代理。",
                    "credit_default": "已可用，作为信用尾部风险基础变量。",
                }[dataset],
            }
        )

    usable_df = pd.DataFrame(usable)
    gaps_df = pd.DataFrame(gaps)
    with pd.ExcelWriter(REPORTS / "credit_tail_gap_final_report.xlsx", engine="openpyxl") as writer:
        gaps_df.to_excel(writer, sheet_name="gap_summary", index=False)
        usable_df.to_excel(writer, sheet_name="usable_downloaded", index=False)
        if not monthly.empty:
            monthly.to_excel(writer, sheet_name="monthly_data", index=False)

    lines = [
        "# 信用尾部基础数据最终缺口报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 已下载",
        "",
    ]
    for _, r in usable_df.iterrows():
        lines.append(f"- {r['dataset']}: {r['rows']}行，{r['start']} 至 {r['end']}，字段：{r['usable_columns']}；备注：{r['quality_note']}")
    lines.extend(["", "## 缺口判断", ""])
    for _, r in gaps_df.iterrows():
        lines.append(f"- {r['dataset']}: {r['status']}。{r['issue']} 下一步：{r['recommended_action']}")
    (REPORTS / "credit_tail_gap_final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(REPORTS / "credit_tail_gap_final_report.md")
    print(REPORTS / "credit_tail_gap_final_report.xlsx")
    print(gaps_df.to_string(index=False))


if __name__ == "__main__":
    main()
