from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "tushare"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
START_DATE = "20180101"
END_DATE = "20260603"


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path.with_suffix(".parquet"), index=False)
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def fetch_paginated(pro, api: str, limit: int = 2000, sleep: float = 0.12, **kwargs) -> pd.DataFrame:
    frames = []
    offset = 0
    while True:
        df = pro.query(api, limit=limit, offset=offset, **kwargs)
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < limit:
            break
        offset += limit
        time.sleep(sleep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_month_end_from_trade_cal(pro) -> list[str]:
    cal = pro.trade_cal(exchange="SSE", start_date=START_DATE, end_date=END_DATE, is_open="1")
    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    month_end = cal.groupby(cal["cal_date"].dt.to_period("M"))["cal_date"].max().dt.strftime("%Y%m%d")
    return month_end.tolist()


def fetch_cb_month_end_daily(pro, dates: list[str]) -> pd.DataFrame:
    frames = []
    fields = "ts_code,trade_date,close,amount,vol,cb_over_rate"
    for d in dates:
        df = pro.cb_daily(trade_date=d, fields=fields)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(0.12)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_monthly_supply(cb_basic: pd.DataFrame, cb_issue: pd.DataFrame, cb_call: pd.DataFrame, cb_share: pd.DataFrame, cb_daily_me: pd.DataFrame) -> pd.DataFrame:
    months = pd.date_range(pd.to_datetime(START_DATE), pd.to_datetime(END_DATE), freq="ME")
    base = cb_basic.copy()
    for col in ["list_date", "delist_date"]:
        base[col] = pd.to_datetime(base[col], errors="coerce")
    base["issue_size"] = pd.to_numeric(base.get("issue_size"), errors="coerce")
    base["remain_size"] = pd.to_numeric(base.get("remain_size"), errors="coerce")

    share = cb_share.copy()
    if not share.empty:
        share["end_date"] = pd.to_datetime(share["end_date"], errors="coerce")
        share["remain_size"] = pd.to_numeric(share.get("remain_size"), errors="coerce")
        share = share.dropna(subset=["ts_code", "end_date"]).sort_values(["ts_code", "end_date"])

    issue = cb_issue.copy()
    if not issue.empty:
        issue["onl_date"] = pd.to_datetime(issue.get("onl_date"), errors="coerce")
        issue["issue_size"] = pd.to_numeric(issue.get("issue_size"), errors="coerce")

    call = cb_call.copy()
    if not call.empty:
        call["call_date"] = pd.to_datetime(call.get("call_date"), errors="coerce")
        call["call_amount"] = pd.to_numeric(call.get("call_amount"), errors="coerce")
        call["call_vol"] = pd.to_numeric(call.get("call_vol"), errors="coerce")

    daily = cb_daily_me.copy()
    if not daily.empty:
        daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
        daily["month"] = daily["trade_date"].dt.to_period("M").dt.to_timestamp("M")
        daily["close"] = pd.to_numeric(daily.get("close"), errors="coerce")
        daily["cb_over_rate"] = pd.to_numeric(daily.get("cb_over_rate"), errors="coerce")
        daily["amount"] = pd.to_numeric(daily.get("amount"), errors="coerce")

    rows = []
    base_idx = base.set_index("ts_code", drop=False)
    for m in months:
        listed = base[(base["list_date"].notna()) & (base["list_date"] <= m)]
        active = listed[(base["delist_date"].isna()) | (base["delist_date"] > m)]
        active_codes = active["ts_code"].dropna().tolist()

        balance = np.nan
        if active_codes:
            if not share.empty:
                hist = share[(share["ts_code"].isin(active_codes)) & (share["end_date"] <= m)]
                latest = hist.sort_values("end_date").groupby("ts_code").tail(1)[["ts_code", "remain_size"]]
                latest = latest.set_index("ts_code")["remain_size"]
            else:
                latest = pd.Series(dtype=float)
            initial = base_idx.reindex(active_codes)["issue_size"]
            remain = initial.copy()
            remain.update(latest.dropna())
            balance = remain.sum(min_count=1)

        if not issue.empty:
            issue_m = issue[issue["onl_date"].dt.to_period("M") == m.to_period("M")]
            issue_size = issue_m["issue_size"].sum(min_count=1)
            issue_count = issue_m["ts_code"].nunique()
        else:
            issue_size, issue_count = np.nan, 0

        if not call.empty:
            call_m = call[call["call_date"].dt.to_period("M") == m.to_period("M")]
            force = call_m[call_m.get("is_call", "").astype(str).str.contains("是|Y|y|1", regex=True, na=False)]
            call_amount = force["call_amount"].sum(min_count=1)
            call_count = force["ts_code"].nunique()
        else:
            call_amount, call_count = np.nan, 0

        if not daily.empty:
            d = daily[daily["month"] == m]
            price_median = d["close"].median()
            price_avg = d["close"].mean()
            premium_median = d["cb_over_rate"].median()
            turnover_amount = d["amount"].sum(min_count=1)
            low_price_ratio = (d["close"] < 110).mean() if len(d) else np.nan
            high_price_ratio = (d["close"] > 130).mean() if len(d) else np.nan
        else:
            price_median = price_avg = premium_median = turnover_amount = low_price_ratio = high_price_ratio = np.nan

        delisted_count = listed[listed["delist_date"].dt.to_period("M") == m.to_period("M")]["ts_code"].nunique()
        rows.append(
            {
                "month": m,
                "cb_active_count": len(active_codes),
                "cb_balance_remain_size": balance,
                "cb_issue_size": issue_size,
                "cb_issue_count": issue_count,
                "cb_delisted_count": delisted_count,
                "cb_force_redeem_count": call_count,
                "cb_force_redeem_amount": call_amount,
                "cb_price_median": price_median,
                "cb_price_avg": price_avg,
                "cb_premium_median": premium_median,
                "cb_turnover_amount": turnover_amount,
                "low_price_cb_ratio": low_price_ratio,
                "high_price_cb_ratio": high_price_ratio,
                "source": "Tushare cb_basic/cb_issue/cb_call/cb_share/cb_daily",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api()

    cb_basic = pro.cb_basic()
    save_table(cb_basic, RAW / "cb_basic")

    cb_issue = fetch_paginated(pro, "cb_issue")
    save_table(cb_issue, RAW / "cb_issue")

    cb_call = fetch_paginated(pro, "cb_call")
    save_table(cb_call, RAW / "cb_call")

    cb_share = fetch_paginated(pro, "cb_share")
    save_table(cb_share, RAW / "cb_share")

    month_end_dates = to_month_end_from_trade_cal(pro)
    cb_daily_me = fetch_cb_month_end_daily(pro, month_end_dates)
    save_table(cb_daily_me, RAW / "cb_daily_month_end")

    monthly = build_monthly_supply(cb_basic, cb_issue, cb_call, cb_share, cb_daily_me)
    save_table(monthly, PROCESSED / "tushare_cb_supply_valuation_monthly")

    quality = pd.DataFrame(
        [
            {"table": "cb_basic", "rows": len(cb_basic), "start": cb_basic.get("list_date", pd.Series(dtype=str)).min(), "end": cb_basic.get("list_date", pd.Series(dtype=str)).max()},
            {"table": "cb_issue", "rows": len(cb_issue), "start": cb_issue.get("onl_date", pd.Series(dtype=str)).min(), "end": cb_issue.get("onl_date", pd.Series(dtype=str)).max()},
            {"table": "cb_call", "rows": len(cb_call), "start": cb_call.get("call_date", pd.Series(dtype=str)).min(), "end": cb_call.get("call_date", pd.Series(dtype=str)).max()},
            {"table": "cb_share", "rows": len(cb_share), "start": cb_share.get("end_date", pd.Series(dtype=str)).min(), "end": cb_share.get("end_date", pd.Series(dtype=str)).max()},
            {"table": "cb_daily_month_end", "rows": len(cb_daily_me), "start": cb_daily_me.get("trade_date", pd.Series(dtype=str)).min(), "end": cb_daily_me.get("trade_date", pd.Series(dtype=str)).max()},
            {"table": "tushare_cb_supply_valuation_monthly", "rows": len(monthly), "start": monthly["month"].min(), "end": monthly["month"].max()},
        ]
    )
    with pd.ExcelWriter(REPORTS / "tushare_cb_data_quality_report.xlsx", engine="openpyxl") as writer:
        quality.to_excel(writer, sheet_name="coverage", index=False)
        monthly.tail(24).to_excel(writer, sheet_name="monthly_tail", index=False)
        monthly.isna().mean().rename("missing_ratio").reset_index().to_excel(writer, sheet_name="missing_ratio", index=False)

    lines = [
        "# Tushare 可转债数据下载报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 输出",
        "- data_raw/tushare/cb_basic.parquet",
        "- data_raw/tushare/cb_issue.parquet",
        "- data_raw/tushare/cb_call.parquet",
        "- data_raw/tushare/cb_share.parquet",
        "- data_raw/tushare/cb_daily_month_end.parquet",
        "- data_processed/tushare_cb_supply_valuation_monthly.parquet",
        "- output/reports/tushare_cb_data_quality_report.xlsx",
        "",
        "## 覆盖",
    ]
    for _, r in quality.iterrows():
        lines.append(f"- {r['table']}: {r['rows']}行, {r['start']} 至 {r['end']}")
    (REPORTS / "tushare_cb_data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(quality.to_string(index=False))
    print(monthly.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
