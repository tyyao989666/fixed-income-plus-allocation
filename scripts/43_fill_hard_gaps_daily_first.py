from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
RAW = ROOT / "data_raw" / "wind" / "hard_gap_fill"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
START = "20180101"
END = "20260604"


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    return [x.strip() for x in raw.split(";") if x.strip()] + [None]


def call_wind(server: str, tool: str, params: dict, cache: Path) -> tuple[object, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8")), "ok_cached"
        except Exception:
            cache.unlink(missing_ok=True)
    last_error: object = None
    for key in key_pool():
        env = os.environ.copy()
        if key:
            env["WIND_API_KEY"] = key
        res = subprocess.run(
            ["node", str(SKILL_CLI), "call", server, tool, json.dumps(params, ensure_ascii=False, separators=(",", ":"))],
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
            last_error = {"error": f"{exc}: {res.stdout[:300]}"}
            continue
        if payload.get("error"):
            last_error = {"error": payload["error"]}
            if "BALANCE_INSUFFICIENT" in json.dumps(payload["error"], ensure_ascii=False):
                continue
            return last_error, "failed"
        data = payload.get("data")
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, "ok"
    return last_error or {"error": "all_keys_failed"}, "failed"


def blocks_to_df(data: object) -> pd.DataFrame:
    blocks = data.get("data", []) if isinstance(data, dict) else data
    frames = []
    if not isinstance(blocks, list):
        return pd.DataFrame()
    for block in blocks:
        if not isinstance(block, dict) or "columns" not in block or "rows" not in block:
            continue
        rows = block.get("rows") or []
        if not rows:
            continue
        cols = [str(c.get("name", f"col_{i}")) for i, c in enumerate(block.get("columns") or [])]
        max_len = max(len(r) for r in rows)
        cols = (cols + [f"col_{i}" for i in range(len(cols), max_len)])[:max_len]
        df = pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=cols)
        df["resolved_question"] = block.get("resolved_question", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def chunks(freq: str) -> list[tuple[str, str, str]]:
    if freq == "quarter":
        periods = pd.period_range("2018Q1", "2026Q2", freq="Q")
        return [(p.strftime("%YQ%q"), p.start_time.strftime("%Y%m%d"), min(p.end_time, pd.Timestamp(END)).strftime("%Y%m%d")) for p in periods]
    out = []
    for year in range(2018, 2027):
        for h, start, end in [("H1", f"{year}0101", f"{year}0630"), ("H2", f"{year}0701", f"{year}1231")]:
            if start <= END:
                out.append((f"{year}{h}", start, min(end, END)))
    return out


def find_col(df: pd.DataFrame, patterns: list[str], numeric: bool | None = None) -> str | None:
    for c in df.columns:
        s = str(c)
        if all(p.lower() in s.lower() for p in patterns):
            if numeric is True and not pd.to_numeric(df[c], errors="coerce").notna().any():
                continue
            return c
    return None


def coalesce_cols(df: pd.DataFrame, patterns: list[str], numeric: bool = False, exclude: list[str] | None = None) -> pd.Series:
    exclude = exclude or []
    cols = []
    for c in df.columns:
        s = str(c)
        if any(x in s for x in exclude):
            continue
        if all(p.lower() in s.lower() for p in patterns):
            if numeric and not pd.to_numeric(df[c], errors="coerce").notna().any():
                continue
            cols.append(c)
    if not cols:
        return pd.Series([pd.NA] * len(df), index=df.index)
    out = pd.Series([pd.NA] * len(df), index=df.index)
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce") if numeric else df[c]
        out = out.combine_first(vals)
    return out


def coalesce_date(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if "日期" in str(c) or "交易时间" in str(c)]
    out = pd.Series([pd.NaT] * len(df), index=df.index)
    for c in cols:
        vals = pd.to_datetime(df[c], errors="coerce")
        out = out.combine_first(vals)
    return out


def pick_value_col(df: pd.DataFrame, factor: str) -> str | None:
    preferred = {
        "us_10y_yield": ["国债收益率:10年", "10年"],
        "usd_index": ["美元指数"],
        "nanhua_commodity": ["收盘价", "每日收盘价"],
        "gold_au9999": ["Au9999", "AU9999", "收盘价"],
    }
    for pat in preferred.get(factor, []):
        c = find_col(df, [pat], numeric=True)
        if c:
            return c
    candidates = []
    for c in df.columns:
        s = str(c)
        if any(x in s for x in ["日期", "代码", "名称", "币种", "question", "factor"]):
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            candidates.append(c)
    return candidates[-1] if candidates else None


def daily_to_monthly_wide(daily: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    tmp = daily.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"]).sort_values("date")
    rows = []
    for m, g in tmp.groupby(tmp["date"].dt.to_period("M")):
        last = g.iloc[-1]
        row = {"month": m.to_timestamp("M")}
        for c in value_cols:
            if c in g.columns:
                row[c] = last.get(c)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")


def build_index_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = {
        "csi1000": ("中证1000", "000852.SH"),
        "dividend": ("中证红利", "000922.CSI"),
        "csi_convertible_bond": ("中证转债", "000832.CSI"),
    }
    frames, logs = [], []
    for name, (cn, code) in targets.items():
        raw_frames = []
        for tag, start, end in chunks("quarter"):
            q = f"查询{start}至{end}{cn}指数{code}每日PE、PB、收盘价、成交额，返回日期、PE、PB、收盘价、成交额。"
            data, status = call_wind("index_data", "get_index_fundamentals", {"question": q, "lang": "中文"}, RAW / "index_valuation" / f"{name}_{tag}.json")
            df = blocks_to_df(data)
            logs.append({"dataset": "index_valuation_daily", "name": name, "chunk": tag, "status": status, "rows": len(df)})
            if not df.empty:
                raw_frames.append(df)
        raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
        raw.to_parquet(PROCESSED / f"raw_wind_{name}_valuation_daily.parquet", index=False)
        if raw.empty:
            continue
        date_vals = coalesce_date(raw)
        if date_vals.isna().all():
            continue
        out = pd.DataFrame({"date": date_vals})
        out["index_name"] = name
        out["wind_code"] = code
        out["pe"] = coalesce_cols(raw, ["PE"], numeric=True)
        out["pb"] = coalesce_cols(raw, ["PB"], numeric=True)
        out["close"] = coalesce_cols(raw, ["收盘"], numeric=True)
        out["amount"] = coalesce_cols(raw, ["成交额"], numeric=True)
        out = out.dropna(subset=["date"]).drop_duplicates(["date", "index_name"]).sort_values(["index_name", "date"])
        out["source"] = "wind_index_fundamentals_quarterly_chunks"
        frames.append(out)
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), pd.DataFrame(logs)


def build_alt_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = {
        "us_10y_yield": "查询{start}至{end}美国10年期国债收益率日频数据，返回日期和值。",
        "usd_index": "查询{start}至{end}美元指数日频收盘价，返回日期和值。",
        "nanhua_commodity": "查询{start}至{end}南华商品指数日频收盘价，返回日期和值。",
        "gold_au9999": "查询{start}至{end}上海黄金交易所Au9999日频收盘价，返回日期和值。",
    }
    frames, logs = [], []
    for factor, template in questions.items():
        raw_frames = []
        for tag, start, end in chunks("half"):
            q = template.format(start=start, end=end)
            data, status = call_wind("analytics_data", "get_financial_data", {"question": q, "lang": "中文"}, RAW / "alt_overseas" / f"{factor}_{tag}.json")
            df = blocks_to_df(data)
            logs.append({"dataset": "alt_overseas_daily", "name": factor, "chunk": tag, "status": status, "rows": len(df)})
            if not df.empty:
                raw_frames.append(df)
        raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
        raw.to_parquet(PROCESSED / f"raw_wind_{factor}_daily.parquet", index=False)
        if raw.empty:
            continue
        date_vals = coalesce_date(raw)
        if date_vals.isna().all():
            continue
        if factor == "us_10y_yield":
            vals = coalesce_cols(raw, ["10年"], numeric=True)
        elif factor == "usd_index":
            vals = coalesce_cols(raw, ["美元指数"], numeric=True)
        elif factor == "nanhua_commodity":
            vals = coalesce_cols(raw, ["收盘价"], numeric=True)
            if vals.isna().all():
                vals = coalesce_cols(raw, ["每日收盘价"], numeric=True)
        elif factor == "gold_au9999":
            vals = coalesce_cols(raw, ["Au9999"], numeric=True)
            if vals.isna().all():
                vals = coalesce_cols(raw, ["AU9999"], numeric=True)
            vals = vals.combine_first(coalesce_cols(raw, ["收盘价"], numeric=True))
        else:
            value_col = pick_value_col(raw, factor)
            vals = pd.to_numeric(raw[value_col], errors="coerce") if value_col else pd.Series([pd.NA] * len(raw), index=raw.index)
        out = pd.DataFrame({"date": date_vals, factor: vals})
        out = out.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        frames.append(out)
    merged = None
    for f in frames:
        merged = f if merged is None else merged.merge(f, on="date", how="outer")
    if merged is None:
        return pd.DataFrame(), pd.DataFrame(logs)
    merged = merged.sort_values("date")
    for c in [x for x in merged.columns if x != "date"]:
        merged[f"{c}_ret_1d"] = merged[c].pct_change(fill_method=None)
    merged["source"] = "wind_analytics_half_year_chunks"
    return merged, pd.DataFrame(logs)


def build_aaa_cd_proxy_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    logs, frames = [], []
    raw_path = PROCESSED / "raw_chinabond_yield_curves.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
        logs.append({"dataset": "aaa_cd_proxy_daily", "source": "akshare_chinabond", "year": "cache", "status": "ok_cached", "rows": len(raw)})
    else:
        for year in range(2018, 2027):
            try:
                df = ak.bond_china_yield(start_date=f"{year}0101", end_date=min(f"{year}1231", END))
                logs.append({"dataset": "aaa_cd_proxy_daily", "source": "akshare_chinabond", "year": year, "status": "ok", "rows": len(df)})
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logs.append({"dataset": "aaa_cd_proxy_daily", "source": "akshare_chinabond", "year": year, "status": f"failed:{exc}", "rows": 0})
        raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        raw.to_parquet(raw_path, index=False)
    out = pd.DataFrame()
    if not raw.empty and "曲线名称" in raw.columns:
        bank = raw[raw["曲线名称"].astype(str).str.contains("商业银行普通债收益率曲线\\(AAA\\)", regex=True)].copy()
        if not bank.empty:
            out = pd.DataFrame({
                "date": pd.to_datetime(bank["日期"], errors="coerce"),
                "aaa_cd_1y_proxy": pd.to_numeric(bank.get("1年"), errors="coerce"),
                "aaa_cd_proxy_source": "akshare_chinabond_commercial_bank_aaa_1y",
            }).dropna(subset=["date"]).sort_values("date")
    shibor_path = ROOT / "data_raw" / "tushare" / "shibor_shibor_20180101_20260603.parquet"
    if shibor_path.exists():
        sh = pd.read_parquet(shibor_path)
        sh_daily = pd.DataFrame({
            "date": pd.to_datetime(sh["date"], errors="coerce"),
            "shibor_3m_proxy": pd.to_numeric(sh.get("3m"), errors="coerce"),
            "shibor_1y_proxy": pd.to_numeric(sh.get("1y"), errors="coerce"),
        }).dropna(subset=["date"]).sort_values("date")
        out = sh_daily if out.empty else out.merge(sh_daily, on="date", how="outer")
    if not out.empty:
        out["aaa_cd_1y_final"] = out.get("aaa_cd_1y_proxy").combine_first(out.get("shibor_1y_proxy"))
        out["source"] = "daily_proxy:chinabond_commercial_bank_aaa_1y; fallback:tushare_shibor_1y"
        out["note"] = "未取得中债商业银行同业存单AAA:1年完整真值；日频保留代理口径。"
        out = out.sort_values("date")
    return out, pd.DataFrame(logs)


def build_rating_proxy_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = "统计2018年1月至2026年6月信用债主体评级下调次数和债项评级下调次数，按月份汇总，返回月份、主体评级下调次数、债项评级下调次数。"
    data, status = call_wind("analytics_data", "get_financial_data", {"question": q, "lang": "中文"}, RAW / "rating_downgrade" / "rating_downgrade_monthly_probe.json")
    probe = blocks_to_df(data)
    logs = [{"dataset": "rating_downgrade", "status": status, "rows": len(probe), "note": "actual monthly/daily table unavailable" if probe.empty else ""}]
    tail_path = PROCESSED / "wind_credit_tail_monthly.parquet"
    if not tail_path.exists():
        return pd.DataFrame(), pd.DataFrame(logs)
    tail = pd.read_parquet(tail_path)
    tail["month"] = pd.to_datetime(tail["month"] if "month" in tail.columns else tail.iloc[:, 0], errors="coerce")
    nums = [c for c in ["cancel_issuance_count", "cancel_issuance_amount", "credit_default_count", "credit_default_amount"] if c in tail.columns]
    out = tail[["month"] + nums].dropna(subset=["month"]).copy()
    for c in nums:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out = out.groupby("month", as_index=False)[nums].sum()
    if nums:
        base = sum(out[c] for c in nums)
        roll = base.rolling(24, min_periods=6)
        out["rating_downgrade_pressure_proxy"] = (base - roll.mean()) / roll.std().replace(0, pd.NA)
    out["actual_rating_downgrade_count"] = pd.NA
    out["source"] = "monthly_proxy_from_cancel_issuance_and_default; actual_rating_downgrade_missing"
    out["note"] = "未取得可靠日频/月频评级下调真值，禁止当作真实评级下调次数使用。"
    return out.sort_values("month"), pd.DataFrame(logs)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    index_daily, index_log = build_index_daily()
    alt_daily, alt_log = build_alt_daily()
    cd_daily, cd_log = build_aaa_cd_proxy_daily()
    rating_monthly, rating_log = build_rating_proxy_monthly()

    index_daily.to_parquet(PROCESSED / "index_valuation_filled_daily.parquet", index=False)
    alt_daily.to_parquet(PROCESSED / "alt_overseas_filled_daily.parquet", index=False)
    cd_daily.to_parquet(PROCESSED / "aaa_cd_1y_proxy_daily.parquet", index=False)
    rating_monthly.to_parquet(PROCESSED / "rating_downgrade_proxy_monthly.parquet", index=False)

    index_monthly = index_daily.pivot_table(index="date", columns="index_name", values=["pe", "pb", "close", "amount"], aggfunc="last").reset_index() if not index_daily.empty else pd.DataFrame()
    if not index_monthly.empty:
        index_monthly.columns = ["date"] + [f"{idx}_{val}" for val, idx in index_monthly.columns[1:]]
        index_monthly = daily_to_monthly_wide(index_monthly.rename(columns={"date": "date"}), [c for c in index_monthly.columns if c != "date"])
    alt_monthly = daily_to_monthly_wide(alt_daily, [c for c in alt_daily.columns if c not in ["date", "source"]]) if not alt_daily.empty else pd.DataFrame()
    cd_monthly = daily_to_monthly_wide(cd_daily, [c for c in cd_daily.columns if c not in ["date", "source", "note", "aaa_cd_proxy_source"]]) if not cd_daily.empty else pd.DataFrame()
    index_monthly.to_parquet(PROCESSED / "index_valuation_filled_monthly.parquet", index=False)
    alt_monthly.to_parquet(PROCESSED / "alt_overseas_filled_monthly.parquet", index=False)
    cd_monthly.to_parquet(PROCESSED / "aaa_cd_1y_proxy_monthly.parquet", index=False)

    logs = pd.concat([index_log, alt_log, cd_log, rating_log], ignore_index=True, sort=False)
    logs.to_csv(REPORTS / "hard_gap_daily_first_fill_log.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(REPORTS / "hard_gap_daily_first_fill_report.xlsx", engine="openpyxl") as writer:
        index_daily.tail(200).to_excel(writer, sheet_name="index_daily_tail", index=False)
        alt_daily.tail(200).to_excel(writer, sheet_name="alt_daily_tail", index=False)
        cd_daily.tail(200).to_excel(writer, sheet_name="aaa_cd_daily_tail", index=False)
        rating_monthly.to_excel(writer, sheet_name="rating_proxy_monthly", index=False)
        logs.to_excel(writer, sheet_name="fetch_log", index=False)
    summary = pd.DataFrame([
        {"table": "index_valuation_filled_daily", "rows": len(index_daily), "start": index_daily["date"].min() if "date" in index_daily else None, "end": index_daily["date"].max() if "date" in index_daily else None},
        {"table": "alt_overseas_filled_daily", "rows": len(alt_daily), "start": alt_daily["date"].min() if "date" in alt_daily else None, "end": alt_daily["date"].max() if "date" in alt_daily else None},
        {"table": "aaa_cd_1y_proxy_daily", "rows": len(cd_daily), "start": cd_daily["date"].min() if "date" in cd_daily else None, "end": cd_daily["date"].max() if "date" in cd_daily else None},
        {"table": "rating_downgrade_proxy_monthly", "rows": len(rating_monthly), "start": rating_monthly["month"].min() if "month" in rating_monthly else None, "end": rating_monthly["month"].max() if "month" in rating_monthly else None},
    ])
    (REPORTS / "hard_gap_daily_first_fill_report.md").write_text(
        "# 日频优先缺口补数报告\n\n"
        + summary.to_markdown(index=False)
        + "\n\n评级下调未取得可靠日频/月频真值，仅保留代理压力表；其余能日频的均已按日频输出，并派生月频。\n\n"
        + "数据来源于万得 Wind 金融数据服务；代理表同时使用 AkShare/中债公开曲线及 Tushare Shibor。\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
