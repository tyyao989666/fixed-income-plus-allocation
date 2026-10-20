from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


def num(x):
    return pd.to_numeric(x, errors="coerce")


def first_numeric(row: pd.Series, patterns: list[str]):
    for col in row.index:
        name = str(col)
        if all(p in name for p in patterns):
            v = num(row[col])
            if pd.notna(v):
                return float(v)
    return None


def main() -> None:
    raw_path = PROCESSED / "wind_credit_tail_raw_tables.parquet"
    raw = pd.read_parquet(raw_path)
    rows = []
    for _, r in raw.iterrows():
        dataset = r["dataset"]
        if dataset in {"credit_net_financing", "cancel_issuance"}:
            y = num(r.get("年份"))
            m = num(r.get("月份"))
        elif dataset == "credit_default":
            y = num(r.get("违约年份"))
            m = num(r.get("违约月份"))
        else:
            continue
        if pd.isna(y) or pd.isna(m):
            continue
        out = {"month": pd.Timestamp(int(y), int(m), 1) + pd.offsets.MonthEnd(0), "dataset": dataset, "data_source": "wind_analytics"}
        if dataset == "credit_net_financing":
            out["credit_bond_issuance_amount"] = first_numeric(r, ["发行", "总额"])
            out["credit_bond_repayment_amount"] = first_numeric(r, ["偿还"])
            out["credit_bond_net_financing"] = first_numeric(r, ["净融资"])
            out["quality_flag"] = "annual_aggregated_not_monthly_reliable"
        elif dataset == "cancel_issuance":
            out["cancel_issuance_count"] = first_numeric(r, ["发行", "只数"])
            out["cancel_issuance_amount"] = first_numeric(r, ["发行", "总额"])
            out["quality_flag"] = "ok"
        elif dataset == "credit_default":
            out["credit_default_amount"] = first_numeric(r, ["违约", "余额"]) or first_numeric(r, ["违约", "规模"])
            out["credit_default_count"] = first_numeric(r, ["违约", "只数"])
            out["quality_flag"] = "ok"
        rows.append(out)
    monthly = pd.DataFrame(rows)
    monthly = monthly.sort_values(["dataset", "month"]).reset_index(drop=True)
    monthly.to_parquet(PROCESSED / "wind_credit_tail_monthly.parquet", index=False)
    monthly.to_csv(PROCESSED / "wind_credit_tail_monthly.csv", index=False, encoding="utf-8-sig")

    summary = monthly.groupby("dataset").agg(
        rows=("month", "size"),
        start=("month", "min"),
        end=("month", "max"),
        quality_flags=("quality_flag", lambda x: ",".join(sorted(set(map(str, x))))),
    ).reset_index()
    with pd.ExcelWriter(REPORTS / "wind_credit_tail_reparse_report.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        monthly.to_excel(writer, sheet_name="monthly", index=False)
    lines = [
        "# Wind 信用尾部月频重解析报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for _, r in summary.iterrows():
        lines.append(f"- {r['dataset']}: {r['rows']}行，{r['start']} 至 {r['end']}，quality={r['quality_flags']}")
    (REPORTS / "wind_credit_tail_reparse_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
