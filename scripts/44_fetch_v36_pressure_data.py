from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
RAW = ROOT / "data_raw" / "wind" / "v36_pressure"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"

START_MONTH = os.environ.get("START_MONTH", "2018-01")
END_MONTH = os.environ.get("END_MONTH", "2026-06")
START_DATE = "20180101"
END_DATE = "20260604"

CREDIT_TYPES = ["公司债", "企业债", "中期票据", "短期融资券", "超短期融资券", "定向工具"]
# Wind analytics currently returns no rows for “政策性金融债/政金债” wording,
# while “金融债” returns a stable daily series. Keep the wider label explicit.
RATE_SUPPLY_TYPES = ["国债", "地方政府债", "金融债"]


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    return [x.strip() for x in raw.split(";") if x.strip()] + [None]


def call_wind(question: str, cache: Path) -> tuple[object, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8")), "ok_cached"
        except Exception:
            cache.unlink(missing_ok=True)
    params = {"question": question.replace(" ", ""), "lang": "中文"}
    last: object = None
    for key in key_pool():
        env = os.environ.copy()
        if key:
            env["WIND_API_KEY"] = key
        res = subprocess.run(
            ["node", str(SKILL_CLI), "call", "analytics_data", "get_financial_data", json.dumps(params, ensure_ascii=False, separators=(",", ":"))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            env=env,
        )
        try:
            outer = json.loads(res.stdout)
            payload = json.loads(outer["content"][0]["text"])
        except Exception as exc:
            last = {"error": f"{exc}: {res.stdout[:300]}"}
            continue
        if payload.get("error"):
            last = {"error": payload["error"]}
            if "BALANCE_INSUFFICIENT" in json.dumps(payload["error"], ensure_ascii=False):
                continue
            return last, "failed"
        data = payload.get("data")
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, "ok"
    return last or {"error": "all_keys_failed"}, "failed"


def blocks_to_df(data: object, meta: dict) -> pd.DataFrame:
    blocks = data.get("data", []) if isinstance(data, dict) else data
    frames = []
    if not isinstance(blocks, list):
        return pd.DataFrame()
    for block in blocks:
        if not isinstance(block, dict) or "columns" not in block:
            continue
        rows = block.get("rows") or []
        if not rows:
            continue
        cols = [str(c.get("name", f"col_{i}")) for i, c in enumerate(block.get("columns") or [])]
        max_len = max(len(r) for r in rows)
        cols = (cols + [f"col_{i}" for i in range(len(cols), max_len)])[:max_len]
        df = pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=cols)
        for k, v in meta.items():
            df[k] = v
        df["resolved_question"] = block.get("resolved_question", "")
        df["excel_total_count"] = block.get("excelTotalCount")
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def coalesce_date(df: pd.DataFrame) -> pd.Series:
    out = pd.Series([pd.NaT] * len(df), index=df.index)
    for c in df.columns:
        if "日期" in str(c) or "交易日期" in str(c):
            out = out.combine_first(pd.to_datetime(df[c], errors="coerce"))
    return out


def coalesce_numeric(df: pd.DataFrame, includes: list[str], excludes: list[str] | None = None) -> pd.Series:
    excludes = excludes or []
    out = pd.Series([pd.NA] * len(df), index=df.index)
    for c in df.columns:
        s = str(c)
        if any(x in s for x in excludes):
            continue
        if all(x in s for x in includes):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                out = out.combine_first(vals)
    return out


def month_scoped_numeric(df: pd.DataFrame, includes: list[str], excludes: list[str] | None = None) -> pd.Series:
    """Prefer dynamic Wind value columns whose title matches the queried month."""
    fallback = coalesce_numeric(df, includes, excludes)
    if "month_query" not in df.columns:
        return fallback
    excludes = excludes or []
    months = pd.to_datetime(df["month_query"], errors="coerce")
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    for idx, m in months.dropna().items():
        ym = f"{m.year}年{m.month}月"
        candidates = [
            c for c in df.columns
            if ym in str(c) and all(x in str(c) for x in includes) and not any(x in str(c) for x in excludes)
        ]
        for c in candidates:
            val = pd.to_numeric(pd.Series([df.at[idx, c]]), errors="coerce").iloc[0]
            if pd.notna(val):
                out.at[idx] = float(val)
                break
    return out.combine_first(fallback)


def keep_rows_in_query_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or "month_query" not in df.columns or date_col not in df.columns:
        return df
    month = pd.to_datetime(df["month_query"], errors="coerce")
    start = month.dt.to_period("M").dt.to_timestamp()
    end = start + pd.offsets.MonthEnd(0)
    return df.loc[month.isna() | df[date_col].between(start, end)].copy()


def month_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}年{ts.month}月"


def fetch_credit_turnover_one(month: pd.Timestamp, bond_type: str) -> tuple[pd.DataFrame, dict]:
    label = month_label(month)
    question = f"统计{label}{bond_type}每日成交金额，按交易日期分组，返回交易日期、成交金额。"
    cache = RAW / "credit_turnover" / f"{bond_type}_{month.strftime('%Y%m')}.json"
    data, status = call_wind(question, cache)
    df = blocks_to_df(data, {"month_query": month, "bond_type": bond_type, "dataset": "credit_turnover", "status": status})
    return df, {"dataset": "credit_turnover", "month": month.strftime("%Y%m"), "bond_type": bond_type, "status": status, "rows": len(df)}


def fetch_rate_supply_one(month: pd.Timestamp, bond_type: str, side: str) -> tuple[pd.DataFrame, dict]:
    label = month_label(month)
    if side == "issue":
        question = f"统计{label}{bond_type}每日发行规模，按发行日期分组，返回发行日期、发行规模合计。"
    else:
        question = f"统计{label}{bond_type}每日到期偿还规模，按到期日期分组，返回到期日期、偿还规模合计。"
    cache = RAW / "rate_supply" / f"{side}_{bond_type}_{month.strftime('%Y%m')}.json"
    data, status = call_wind(question, cache)
    df = blocks_to_df(data, {"month_query": month, "bond_type": bond_type, "side": side, "dataset": "rate_supply", "status": status})
    return df, {"dataset": "rate_supply", "month": month.strftime("%Y%m"), "bond_type": bond_type, "side": side, "status": status, "rows": len(df)}


def build_credit_turnover(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": coalesce_date(raw),
            "bond_type": raw["bond_type"],
            "month_query": raw["month_query"],
            "credit_turnover_amount": month_scoped_numeric(raw, ["成交金额"]),
        }
    ).dropna(subset=["date"])
    out = keep_rows_in_query_month(out)
    out["credit_turnover_amount"] = out["credit_turnover_amount"].fillna(0)
    daily = out.groupby("date", as_index=False).agg(credit_turnover_amount=("credit_turnover_amount", "sum"))
    daily = daily.sort_values("date")
    daily["credit_turnover_ma3"] = daily["credit_turnover_amount"].rolling(3, min_periods=3).mean()
    daily["credit_turnover_ma20"] = daily["credit_turnover_amount"].rolling(20, min_periods=10).mean()
    daily["credit_turnover_ratio_3_20"] = daily["credit_turnover_ma3"] / daily["credit_turnover_ma20"]
    daily["credit_turnover_stress"] = (daily["credit_turnover_ratio_3_20"] < 0.70).astype("Int64")
    by_type = out.groupby(["date", "bond_type"], as_index=False).agg(credit_turnover_amount=("credit_turnover_amount", "sum"))
    return daily, by_type


def build_rate_supply(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    amount_issue = month_scoped_numeric(raw, ["发行"], excludes=["偿还", "到期", "净"])
    amount_maturity = month_scoped_numeric(raw, ["偿还"], excludes=["净"])
    tmp = pd.DataFrame({"date": coalesce_date(raw), "bond_type": raw["bond_type"], "side": raw["side"], "month_query": raw["month_query"], "issue": amount_issue, "maturity": amount_maturity}).dropna(subset=["date"])
    tmp = keep_rows_in_query_month(tmp)
    tmp["amount"] = tmp["issue"].where(tmp["side"].eq("issue"), tmp["maturity"])
    tmp["amount"] = pd.to_numeric(tmp["amount"], errors="coerce").fillna(0)
    by_type = tmp.groupby(["date", "bond_type", "side"], as_index=False).agg(amount=("amount", "sum"))
    piv = by_type.pivot_table(index=["date", "bond_type"], columns="side", values="amount", aggfunc="sum").reset_index()
    for c in ["issue", "maturity"]:
        if c not in piv.columns:
            piv[c] = 0.0
    piv[["issue", "maturity"]] = piv[["issue", "maturity"]].fillna(0.0)
    piv["net_supply"] = piv["issue"] - piv["maturity"]
    wide = piv.pivot_table(index="date", columns="bond_type", values=["issue", "maturity", "net_supply"], aggfunc="sum")
    wide.columns = [f"{bt}_{metric}" for metric, bt in wide.columns]
    wide = wide.reset_index().fillna(0)
    issue_cols = [c for c in wide.columns if c.endswith("_issue")]
    maturity_cols = [c for c in wide.columns if c.endswith("_maturity")]
    net_cols = [c for c in wide.columns if c.endswith("_net_supply")]
    wide["rate_bond_issue_total"] = wide[issue_cols].sum(axis=1) if issue_cols else 0
    wide["rate_bond_maturity_total"] = wide[maturity_cols].sum(axis=1) if maturity_cols else 0
    wide["rate_supply_net"] = wide[net_cols].sum(axis=1) if net_cols else 0
    wide = wide.sort_values("date")
    wide["rate_supply_net_20d"] = wide["rate_supply_net"].rolling(20, min_periods=10).sum()
    return wide, piv


def build_tl_futures() -> pd.DataFrame:
    pro = ts.pro_api(ts.get_token())
    basic = pro.fut_basic(exchange="CFFEX")
    tl = basic[basic["fut_code"].eq("TL")].copy()
    frames = []
    for _, r in tl.iterrows():
        code = r["ts_code"]
        start = max(str(r.get("list_date") or START_DATE), START_DATE)
        end = min(str(r.get("delist_date") or END_DATE), END_DATE)
        try:
            df = pro.fut_daily(ts_code=code, start_date=start, end_date=end)
            if not df.empty:
                df["contract"] = code
                frames.append(df)
            time.sleep(0.08)
        except Exception:
            continue
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    raw.to_parquet(PROCESSED / "raw_tushare_tl_futures_daily.parquet", index=False)
    if raw.empty:
        return pd.DataFrame()
    raw["date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    for c in ["vol", "amount", "oi", "close", "settle"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    agg = raw.groupby("date", as_index=False).agg(tl_volume=("vol", "sum"), tl_amount=("amount", "sum"), tl_oi=("oi", "sum"))
    active = raw.sort_values(["date", "oi"], ascending=[True, False]).drop_duplicates("date")
    active = active[["date", "contract", "close", "settle", "oi"]].rename(columns={"contract": "tl_active_contract", "close": "tl_active_close", "settle": "tl_active_settle", "oi": "tl_active_oi"})
    out = agg.merge(active, on="date", how="left").sort_values("date")
    out["tl_volume_oi_ratio"] = out["tl_volume"] / out["tl_oi"].replace(0, pd.NA)
    out["source"] = "tushare_fut_daily_30y_treasury_futures_TL"
    return out


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(START_MONTH, END_MONTH, freq="M").to_timestamp()
    workers = int(os.environ.get("WIND_WORKERS", "4"))
    logs = []
    credit_frames = []
    supply_frames = []

    credit_jobs = [(m, bt) for m in months for bt in CREDIT_TYPES]
    supply_jobs = [(m, bt, side) for m in months for bt in RATE_SUPPLY_TYPES for side in ["issue", "maturity"]]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_credit_turnover_one, *j) for j in credit_jobs]
        futs += [ex.submit(fetch_rate_supply_one, *j) for j in supply_jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            df, log = fut.result()
            logs.append(log)
            if log["dataset"] == "credit_turnover" and not df.empty:
                credit_frames.append(df)
            if log["dataset"] == "rate_supply" and not df.empty:
                supply_frames.append(df)
            if i % 50 == 0:
                print(f"finished {i}/{len(futs)}")

    credit_raw = pd.concat(credit_frames, ignore_index=True, sort=False) if credit_frames else pd.DataFrame()
    supply_raw = pd.concat(supply_frames, ignore_index=True, sort=False) if supply_frames else pd.DataFrame()
    credit_daily, credit_by_type = build_credit_turnover(credit_raw)
    supply_daily, supply_by_type = build_rate_supply(supply_raw)
    tl_daily = build_tl_futures()

    credit_raw.to_parquet(PROCESSED / "raw_wind_credit_turnover_daily.parquet", index=False)
    credit_daily.to_parquet(PROCESSED / "credit_turnover_daily.parquet", index=False)
    credit_by_type.to_parquet(PROCESSED / "credit_turnover_by_type_daily.parquet", index=False)
    supply_raw.to_parquet(PROCESSED / "raw_wind_rate_supply_daily.parquet", index=False)
    supply_daily.to_parquet(PROCESSED / "rate_supply_daily.parquet", index=False)
    supply_by_type.to_parquet(PROCESSED / "rate_supply_by_type_daily.parquet", index=False)
    tl_daily.to_parquet(PROCESSED / "tl_futures_daily.parquet", index=False)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(REPORTS / "v36_pressure_data_fetch_log.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(REPORTS / "v36_pressure_data_report.xlsx", engine="openpyxl") as writer:
        credit_daily.tail(250).to_excel(writer, sheet_name="credit_turnover_tail", index=False)
        supply_daily.tail(250).to_excel(writer, sheet_name="rate_supply_tail", index=False)
        tl_daily.tail(250).to_excel(writer, sheet_name="tl_futures_tail", index=False)
        log_df.to_excel(writer, sheet_name="fetch_log", index=False)
    summary = pd.DataFrame(
        [
            {"table": "credit_turnover_daily", "rows": len(credit_daily), "start": credit_daily["date"].min() if "date" in credit_daily else None, "end": credit_daily["date"].max() if "date" in credit_daily else None},
            {"table": "rate_supply_daily", "rows": len(supply_daily), "start": supply_daily["date"].min() if "date" in supply_daily else None, "end": supply_daily["date"].max() if "date" in supply_daily else None},
            {"table": "tl_futures_daily", "rows": len(tl_daily), "start": tl_daily["date"].min() if "date" in tl_daily else None, "end": tl_daily["date"].max() if "date" in tl_daily else None},
        ]
    )
    (REPORTS / "v36_pressure_data_report.md").write_text(
        "# v3.6 压力模式补数报告\n\n" + summary.to_markdown(index=False) + "\n\n数据来源于万得 Wind 金融数据服务；TL期货使用Tushare中金所30年国债期货日行情。\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
