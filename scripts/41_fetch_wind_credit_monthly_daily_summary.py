from __future__ import annotations

import calendar
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
RAW = ROOT / "data_raw" / "wind" / "credit_monthly_daily_summary"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"

BOND_TYPES = ["公司债", "企业债", "中期票据", "短期融资券", "超短期融资券", "定向工具"]


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    keys = [x.strip() for x in raw.split(";") if x.strip()]
    return keys + [None]


def month_label(month: pd.Timestamp) -> str:
    return f"{month.year}年{month.month}月"


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
        res = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            env=env,
        )
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


def payload_to_frame(payload: object, dataset: str, bond_type: str, month: pd.Timestamp, status: str) -> pd.DataFrame:
    blocks = payload.get("data", []) if isinstance(payload, dict) else payload
    frames: list[pd.DataFrame] = []
    if not isinstance(blocks, list):
        return pd.DataFrame()
    for i, block in enumerate(blocks):
        if not isinstance(block, dict) or "columns" not in block or "rows" not in block:
            continue
        rows = block.get("rows") or []
        if not rows:
            continue
        cols = [str(c.get("name", f"col_{j}")) for j, c in enumerate(block.get("columns") or [])]
        max_len = max(len(r) for r in rows)
        cols = (cols + [f"col_{j}" for j in range(len(cols), max_len)])[:max_len]
        df = pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=cols)
        df.insert(0, "dataset", dataset)
        df.insert(1, "bond_type_query", bond_type)
        df.insert(2, "query_month", month)
        df.insert(3, "table_no", i)
        df["fetch_status"] = status
        df["excel_total_count"] = block.get("excelTotalCount")
        df["resolved_question"] = block.get("resolved_question", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def find_date_col(df: pd.DataFrame) -> str | None:
    preferred = ["发行日期", "发行起始日期", "到期日期", "日期"]
    for name in preferred:
        if name in df.columns and df[name].notna().any():
            return name
    for col in df.columns:
        if "日期" in str(col) and df[col].notna().any():
            return col
    return None


def numeric_candidates(df: pd.DataFrame) -> list[str]:
    skip = {
        "dataset",
        "bond_type_query",
        "query_month",
        "table_no",
        "fetch_status",
        "excel_total_count",
        "resolved_question",
    }
    out = []
    for col in df.columns:
        s = str(col)
        if col in skip or "日期" in s or "板块" in s or "类型" in s or "question" in s or "status" in s:
            continue
        if "只数" in s or "数量" in s:
            continue
        if not any(x in s for x in ["规模", "总额", "偿还量", "兑付", "amount"]):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            out.append(col)
    return out


def count_candidates(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        s = str(col)
        if "只数" in s or "数量" in s:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                out.append(col)
    return out


def standardize(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if raw.empty:
        return raw, pd.DataFrame()

    rows = []
    for (dataset, bond_type, query_month), g in raw.groupby(["dataset", "bond_type_query", "query_month"], dropna=False):
        date_col = find_date_col(g)
        amount_cols = numeric_candidates(g)
        count_cols = count_candidates(g)
        amount_col = amount_cols[0] if amount_cols else None
        count_col = count_cols[0] if count_cols else None
        if not date_col or not amount_col:
            rows.append(
                {
                    "month": query_month,
                    "dataset": dataset,
                    "bond_type": bond_type,
                    "amount": 0.0,
                    "count": 0.0,
                    "row_count": len(g),
                    "quality_flag": "no_parseable_amount",
                }
            )
            continue
        tmp = g.copy()
        tmp["date"] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp["amount"] = pd.to_numeric(tmp[amount_col], errors="coerce")
        tmp["count"] = pd.to_numeric(tmp[count_col], errors="coerce") if count_col else 0
        tmp = tmp[tmp["date"].notna()]
        target_month = pd.Timestamp(query_month).to_period("M")
        tmp = tmp[tmp["date"].dt.to_period("M").eq(target_month)]
        rows.append(
            {
                "month": pd.Timestamp(query_month).to_period("M").to_timestamp("M"),
                "dataset": dataset,
                "bond_type": bond_type,
                "amount": float(tmp["amount"].sum()),
                "count": float(tmp["count"].sum()) if count_col else float(len(tmp)),
                "row_count": int(len(tmp)),
                "amount_col": amount_col,
                "count_col": count_col,
                "quality_flag": "ok" if len(tmp) else "empty",
            }
        )
    by_type = pd.DataFrame(rows)
    piv_amount = by_type.pivot_table(index="month", columns="dataset", values="amount", aggfunc="sum").reset_index()
    piv_count = by_type.pivot_table(index="month", columns="dataset", values="count", aggfunc="sum").reset_index()
    out = piv_amount.rename(columns={"issue": "credit_bond_issuance_amount", "maturity": "credit_bond_maturity_amount"})
    counts = piv_count.rename(columns={"issue": "credit_bond_issuance_count", "maturity": "credit_bond_maturity_count"})
    out = out.merge(counts, on="month", how="outer")
    for col in [
        "credit_bond_issuance_amount",
        "credit_bond_maturity_amount",
        "credit_bond_issuance_count",
        "credit_bond_maturity_count",
    ]:
        if col not in out:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    out["credit_bond_net_financing"] = out["credit_bond_issuance_amount"] - out["credit_bond_maturity_amount"]
    out["data_source"] = "wind_bond_data_monthly_daily_summary"
    out["quality_flag"] = "monthly_daily_grouped"
    out = out.sort_values("month")
    return by_type.sort_values(["month", "dataset", "bond_type"]), out


def task(month: pd.Timestamp, bond_type: str, dataset: str) -> tuple[pd.DataFrame, dict]:
    ym = month.strftime("%Y%m")
    if dataset == "issue":
        if bond_type == "短期融资券":
            question = f"统计{month_label(month)}一般短期融资券发行明细按发行起始日期汇总，不包含SCP，返回发行起始日期、发行总额合计、发行只数。"
        elif bond_type == "超短期融资券":
            question = f"统计{month_label(month)}超短期融资券发行明细按发行起始日期汇总，返回发行起始日期、发行总额合计、发行只数。"
        elif bond_type == "定向工具":
            question = f"统计{month_label(month)}PPN发行明细按发行起始日期汇总，返回发行起始日期、发行总额合计、发行只数。"
        else:
            question = f"统计{month_label(month)}{bond_type}每日发行规模和发行只数，按发行日期分组，返回日期、发行规模合计、发行只数。"
    else:
        if bond_type == "短期融资券":
            phrase = "CP债券到期兑付"
            question = f"统计{month_label(month)}{phrase}规模和只数，按到期日期分组，返回日期、偿还规模合计、只数。"
        elif bond_type == "超短期融资券":
            question = f"统计{month_label(month)}{bond_type}每日到期偿还规模和到期只数，按到期日期分组，返回日期、偿还规模合计、到期只数。"
        else:
            question = f"统计{month_label(month)}{bond_type}每日到期兑付规模和到期只数，按到期日期分组，返回日期、偿还规模合计、到期只数。"
    cache = RAW / f"{dataset}_{bond_type}_{ym}.json"
    payload, status = run_wind(question, cache)
    df = payload_to_frame(payload, dataset, bond_type, month, status)
    return df, {"month": ym, "bond_type": bond_type, "dataset": dataset, "status": status, "rows": len(df), "question": question}


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    start = os.environ.get("START_MONTH", "2025-01")
    end = os.environ.get("END_MONTH", "2026-06")
    months = pd.period_range(start, end, freq="M").to_timestamp()
    jobs = [(m, bt, ds) for m in months for bt in BOND_TYPES for ds in ["issue", "maturity"]]
    frames: list[pd.DataFrame] = []
    logs: list[dict] = []
    workers = int(os.environ.get("WIND_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, *job) for job in jobs]
        for fut in as_completed(futs):
            df, log = fut.result()
            if not df.empty:
                frames.append(df)
            logs.append(log)
            if len(logs) % 20 == 0:
                print(f"finished {len(logs)}/{len(jobs)}")

    by_type, monthly = standardize(frames)
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    raw.to_parquet(PROCESSED / "wind_credit_monthly_daily_summary_raw.parquet", index=False)
    by_type.to_parquet(PROCESSED / "wind_credit_monthly_daily_summary_by_type.parquet", index=False)
    monthly.to_parquet(PROCESSED / "wind_credit_net_financing_monthly_daily_summary.parquet", index=False)
    monthly.to_csv(PROCESSED / "wind_credit_net_financing_monthly_daily_summary.csv", index=False, encoding="utf-8-sig")

    log_df = pd.DataFrame(logs).sort_values(["month", "bond_type", "dataset"])
    with pd.ExcelWriter(REPORTS / "wind_credit_monthly_daily_summary_report.xlsx", engine="openpyxl") as writer:
        monthly.to_excel(writer, sheet_name="monthly", index=False)
        by_type.to_excel(writer, sheet_name="by_type", index=False)
        log_df.to_excel(writer, sheet_name="fetch_log", index=False)

    md = [
        "# Wind信用债月频净融资（日度汇总口径）",
        "",
        f"- 区间：{start} 至 {end}",
        "- 口径：公司债、企业债、中期票据、短期融资券、超短期融资券、定向工具；剔除同业存单和ABS。",
        "- 方法：逐月、逐券种请求 Wind 每日发行/到期兑付汇总，再按月加总，避免明细表100行截断。",
        f"- 月度行数：{len(monthly)}",
        f"- 原始汇总行数：{len(raw)}",
        "",
        "## 最近月份",
        "",
        monthly.tail(12).to_markdown(index=False),
    ]
    (REPORTS / "wind_credit_monthly_daily_summary_report.md").write_text("\n".join(md), encoding="utf-8")
    print(monthly.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
