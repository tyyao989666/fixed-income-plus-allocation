from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
RAW_WIND = ROOT / "data_raw" / "wind"
RAW_TUSHARE = ROOT / "data_raw" / "tushare"
RAW_V42 = RAW_WIND / "v42_gap_fill"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")

START_DATE = "20180101"
END_DATE = "20260604"


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    return [x.strip() for x in raw.split(";") if x.strip()] + [None]


def call_wind(question: str, cache: Path, timeout: int = 180) -> tuple[object, str]:
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
        res = subprocess.run(
            [
                "node",
                str(SKILL_CLI),
                "call",
                "analytics_data",
                "get_financial_data",
                json.dumps(params, ensure_ascii=False, separators=(",", ":")),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        try:
            outer = json.loads(res.stdout)
            payload = json.loads(outer["content"][0]["text"])
        except Exception as exc:
            last_error = {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "stdout": res.stdout[:400]}
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


def blocks_to_df(data: object, meta: dict | None = None) -> pd.DataFrame:
    meta = meta or {}
    blocks = data.get("data", data) if isinstance(data, dict) else data
    frames: list[pd.DataFrame] = []
    if not isinstance(blocks, list):
        return pd.DataFrame()
    for block in blocks:
        if not isinstance(block, dict) or "columns" not in block:
            continue
        rows = block.get("rows") or []
        if not rows:
            continue
        raw_cols = [str(c.get("name", f"col_{i}")) for i, c in enumerate(block.get("columns") or [])]
        max_len = max(len(r) for r in rows)
        raw_cols = (raw_cols + [f"col_{i}" for i in range(len(raw_cols), max_len)])[:max_len]
        seen: dict[str, int] = {}
        cols: list[str] = []
        for c in raw_cols:
            n = seen.get(c, 0)
            cols.append(c if n == 0 else f"{c}__{n + 1}")
            seen[c] = n + 1
        df = pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=cols)
        for k, v in meta.items():
            df[k] = v
        df["resolved_question"] = block.get("resolved_question", "")
        df["excel_total_count"] = block.get("excelTotalCount")
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def coalesce_date(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for c in df.columns:
        s = str(c)
        if "日期" in s or "交易日" in s or "date" in s.lower():
            vals = pd.to_datetime(df[c], errors="coerce")
            out = out.combine_first(vals)
    return out


def numeric_col(df: pd.DataFrame, includes: list[str], excludes: list[str] | None = None) -> pd.Series:
    excludes = excludes or []
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in df.columns:
        s = str(c)
        if any(x in s for x in excludes):
            continue
        if all(x in s for x in includes):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                out = out.combine_first(vals)
    return out


def month_chunks(start: str = START_DATE, end: str = END_DATE) -> list[tuple[str, str, str]]:
    periods = pd.period_range(pd.to_datetime(start), pd.to_datetime(end), freq="M")
    out: list[tuple[str, str, str]] = []
    end_ts = pd.to_datetime(end)
    for p in periods:
        st = p.start_time
        ed = min(p.end_time.normalize(), end_ts)
        out.append((p.strftime("%Y%m"), st.strftime("%Y%m%d"), ed.strftime("%Y%m%d")))
    return out


def quarter_chunks(start: str = START_DATE, end: str = END_DATE) -> list[tuple[str, str, str]]:
    periods = pd.period_range(pd.to_datetime(start), pd.to_datetime(end), freq="Q")
    out: list[tuple[str, str, str]] = []
    end_ts = pd.to_datetime(end)
    for p in periods:
        st = p.start_time
        ed = min(p.end_time.normalize(), end_ts)
        out.append((p.strftime("%YQ%q"), st.strftime("%Y%m%d"), ed.strftime("%Y%m%d")))
    return out


def read_tushare_table(pattern: str) -> pd.DataFrame:
    files = sorted(RAW_TUSHARE.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def period_to_month_end(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.replace("-", "", regex=False).str.slice(0, 6)
    return pd.to_datetime(raw + "01", errors="coerce") + pd.offsets.MonthEnd(0)


def build_macro_monthly() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    cpi = read_tushare_table("cn_cpi*_*.parquet")
    if not cpi.empty and "month" in cpi.columns:
        frames.append(pd.DataFrame({"month": period_to_month_end(cpi["month"]), "cpi_yoy": pd.to_numeric(cpi.get("nt_yoy"), errors="coerce")}))

    ppi = read_tushare_table("cn_ppi*_*.parquet")
    if not ppi.empty and "month" in ppi.columns:
        val_col = "ppi_yoy" if "ppi_yoy" in ppi.columns else ("yoy" if "yoy" in ppi.columns else ppi.columns[-1])
        frames.append(pd.DataFrame({"month": period_to_month_end(ppi["month"]), "ppi_yoy": pd.to_numeric(ppi.get(val_col), errors="coerce")}))

    m2 = read_tushare_table("cn_m*_*.parquet")
    if not m2.empty and "month" in m2.columns:
        col = "m2_yoy" if "m2_yoy" in m2.columns else ("m2_yoy" if "m2_yoy" in m2.columns else None)
        if col is None:
            candidates = [c for c in m2.columns if "m2" in c.lower() and "yoy" in c.lower()]
            col = candidates[0] if candidates else None
        if col:
            frames.append(pd.DataFrame({"month": period_to_month_end(m2["month"]), "m2_yoy": pd.to_numeric(m2[col], errors="coerce")}))

    sf = read_tushare_table("sf_month*_*.parquet")
    if not sf.empty:
        month_col = "month" if "month" in sf.columns else ("date" if "date" in sf.columns else sf.columns[0])
        out = pd.DataFrame({"month": period_to_month_end(sf[month_col])})
        for c in sf.columns:
            lc = c.lower()
            if "inc_month" in lc or "increment" in lc or c in ["inc_month", "incr_amount"]:
                out["social_financing_new"] = pd.to_numeric(sf[c], errors="coerce")
            if "stock" in lc or c in ["end_month", "stock"]:
                out["social_financing_stock"] = pd.to_numeric(sf[c], errors="coerce")
            if "yoy" in lc:
                out["social_financing_yoy"] = pd.to_numeric(sf[c], errors="coerce")
        frames.append(out)

    pmi = read_tushare_table("cn_pmi*_*.parquet")
    if not pmi.empty and "month" in pmi.columns:
        col = "pmi010000" if "pmi010000" in pmi.columns else ("manufacturing_pmi" if "manufacturing_pmi" in pmi.columns else None)
        if col is None:
            candidates = [c for c in pmi.columns if c != "month" and pd.to_numeric(pmi[c], errors="coerce").notna().any()]
            col = candidates[0] if candidates else None
        if col:
            frames.append(pd.DataFrame({"month": period_to_month_end(pmi["month"]), "pmi": pd.to_numeric(pmi[col], errors="coerce")}))

    merged: pd.DataFrame | None = None
    for f in frames:
        f = f.dropna(subset=["month"]).drop_duplicates("month")
        merged = f if merged is None else merged.merge(f, on="month", how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=["month"])
    merged = merged.sort_values("month")

    # Activity data from Wind cache: industrial output, FAI, export. Dates are data periods, not true announcement dates.
    macro_raw = RAW_WIND / "analytics_macro_release_20180101_20260603.json"
    if macro_raw.exists():
        wind_df = blocks_to_df(json.loads(macro_raw.read_text(encoding="utf-8")))
        if not wind_df.empty:
            tmp = pd.DataFrame({"month": coalesce_date(wind_df) + pd.offsets.MonthEnd(0)})
            tmp["industrial_production_yoy"] = numeric_col(wind_df, ["工业", "同比"])
            tmp["fixed_asset_investment_yoy"] = numeric_col(wind_df, ["固定资产", "同比"])
            tmp["export_yoy"] = numeric_col(wind_df, ["出口", "同比"])
            tmp = tmp.groupby("month", as_index=False).last()
            merged = merged.merge(tmp, on="month", how="outer")

    merged["effective_date_conservative"] = merged["month"] + pd.Timedelta(days=25)
    merged["source"] = "tushare_macro_plus_wind_activity_cache"
    merged.to_parquet(PROCESSED / "macro_factors_monthly_v42.parquet", index=False)
    merged.to_csv(PROCESSED / "macro_factors_monthly_v42.csv", index=False, encoding="utf-8-sig")
    return merged


def fetch_monthly_series(kind: str, question_template: str, max_workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs: list[dict] = []
    raw_frames: list[pd.DataFrame] = []

    def one(item: tuple[str, str, str]) -> tuple[pd.DataFrame, dict]:
        tag, start, end = item
        q = question_template.format(start=start, end=end)
        data, status = call_wind(q, RAW_V42 / kind / f"{kind}_{tag}.json")
        df = blocks_to_df(data, {"chunk": tag, "dataset": kind, "status": status})
        return df, {"dataset": kind, "chunk": tag, "status": status, "rows": len(df)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, item) for item in month_chunks()]
        for fut in as_completed(futs):
            df, log = fut.result()
            logs.append(log)
            if not df.empty:
                raw_frames.append(df)
            time.sleep(0.02)

    raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
    return raw, pd.DataFrame(logs)


def build_funding_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    template = "查询{start}至{end}银行间质押式回购加权利率7天和存款类机构质押式回购加权利率7天日频数据，返回日期、R007、DR007。"
    raw, logs = fetch_monthly_series("funding_dr007_r007", template, max_workers=5)
    cache = RAW_WIND / "analytics_dr007_r007_20180101_20260603.json"
    if cache.exists():
        cached = blocks_to_df(json.loads(cache.read_text(encoding="utf-8")), {"chunk": "old_full_query_cache", "dataset": "funding_dr007_r007", "status": "ok_cached"})
        raw = pd.concat([raw, cached], ignore_index=True, sort=False) if not cached.empty else raw
    raw.to_parquet(PROCESSED / "raw_wind_funding_dr007_r007_v42.parquet", index=False)
    if raw.empty:
        return pd.DataFrame(), logs
    out = pd.DataFrame({"date": coalesce_date(raw)})
    out["r007"] = numeric_col(raw, ["回购", "7天"], excludes=["存款类"]).combine_first(numeric_col(raw, ["R007"]))
    out["dr007"] = numeric_col(raw, ["存款类", "7天"]).combine_first(numeric_col(raw, ["DR007"]))
    out = out.dropna(subset=["date"]).groupby("date", as_index=False).last().sort_values("date")
    out["r007_dr007_spread"] = out["r007"] - out["dr007"]
    out["funding_stress_score_proxy"] = (
        out["dr007"].rolling(20, min_periods=10).rank(pct=True) * 0.5
        + out["r007_dr007_spread"].rolling(20, min_periods=10).rank(pct=True) * 0.5
    )
    out["source"] = "wind_dr007_r007_monthly_chunks"
    out.to_parquet(PROCESSED / "funding_rates_daily_v42.parquet", index=False)
    return out, logs


def build_omo_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    template = "查询{start}至{end}公开市场操作货币净投放和7天逆回购利率日频数据，返回日期、公开市场操作货币净投放、7天逆回购利率。"
    raw, logs = fetch_monthly_series("omo_net_and_7d", template, max_workers=4)
    cache = RAW_WIND / "analytics_omo_policy_20180101_20260603.json"
    if cache.exists():
        cached = blocks_to_df(json.loads(cache.read_text(encoding="utf-8")), {"chunk": "old_full_query_cache", "dataset": "omo_net_and_7d", "status": "ok_cached"})
        raw = pd.concat([raw, cached], ignore_index=True, sort=False) if not cached.empty else raw
    raw.to_parquet(PROCESSED / "raw_wind_omo_policy_v42.parquet", index=False)
    if raw.empty:
        return pd.DataFrame(), logs
    out = pd.DataFrame({"date": coalesce_date(raw)})
    out["omo_net_injection"] = numeric_col(raw, ["净投放"])
    out["omo_7d_rate"] = numeric_col(raw, ["7天", "逆回购"], excludes=["净投放"])
    out = out.dropna(subset=["date"]).groupby("date", as_index=False).last().sort_values("date")
    out["omo_net_injection_20d"] = out["omo_net_injection"].rolling(20, min_periods=10).sum()
    out["source"] = "wind_omo_monthly_chunks"
    out.to_parquet(PROCESSED / "omo_policy_daily_v42.parquet", index=False)
    return out, logs


CURVE_SPECS = {
    "cdb": "中债国开债收益率曲线",
    "mtb_aaa": "中债中短期票据收益率曲线(AAA)",
    "mtb_aa_plus": "中债中短期票据收益率曲线(AA+)",
    "mtb_aa": "中债中短期票据收益率曲线(AA)",
}


def fetch_curve_quarters() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    logs: list[dict] = []
    tasks: list[tuple[str, str, str, str]] = []
    for tag, start, end in quarter_chunks():
        for name, cn in CURVE_SPECS.items():
            tasks.append((name, cn, start, end))

    def one(task: tuple[str, str, str, str]) -> tuple[pd.DataFrame, dict]:
        name, cn, start, end = task
        tag = f"{start}_{end}"
        q = f"查询{start}至{end}{cn}1年、3年、5年、10年日频数据，返回日期、1年、3年、5年、10年。"
        data, status = call_wind(q, RAW_V42 / "chinabond_curves" / f"{name}_{tag}.json")
        df = blocks_to_df(data, {"curve_key": name, "curve_name": cn, "status": status, "chunk": tag})
        return df, {"dataset": "chinabond_curve", "curve": name, "chunk": tag, "status": status, "rows": len(df)}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for fut in as_completed(futs):
            df, log = fut.result()
            logs.append(log)
            if not df.empty:
                frames.append(df)
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    raw.to_parquet(PROCESSED / "raw_wind_chinabond_curves_v42.parquet", index=False)
    return raw, pd.DataFrame(logs)


def build_credit_curves_daily(raw: pd.DataFrame) -> pd.DataFrame:
    old = PROCESSED / "raw_chinabond_yield_curves.parquet"
    frames: list[pd.DataFrame] = []
    if old.exists():
        df = pd.read_parquet(old).copy()
        name_map = {
            "中债国债收益率曲线": "treasury",
            "中债中短期票据收益率曲线(AAA)": "mtb_aaa",
            "中债商业银行普通债收益率曲线(AAA)": "bank_bond_aaa",
        }
        df["curve_key"] = df["曲线名称"].map(name_map)
        frames.append(df.rename(columns={"日期": "date", "1年": "y1", "3年": "y3", "5年": "y5", "10年": "y10"})[["date", "curve_key", "y1", "y3", "y5", "y10"]])

    if not raw.empty:
        tmp = pd.DataFrame({"date": coalesce_date(raw), "curve_key": raw["curve_key"]})
        for tenor, pats in {"y1": ["1年"], "y3": ["3年"], "y5": ["5年"], "y10": ["10年"]}.items():
            tmp[tenor] = numeric_col(raw, pats)
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True, sort=False)
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date", "curve_key"]).drop_duplicates(["date", "curve_key"]).sort_values(["date", "curve_key"])
    long.to_parquet(PROCESSED / "credit_yield_curves_daily_v42_long.parquet", index=False)
    wide = long.pivot_table(index="date", columns="curve_key", values=["y1", "y3", "y5", "y10"], aggfunc="last")
    wide.columns = [f"{curve}_{tenor}" for tenor, curve in wide.columns]
    wide = wide.reset_index().sort_values("date")
    for bucket, credit_col, base_col in [
        ("aaa_1_3y", "mtb_aaa_y3", "cdb_y3"),
        ("aa_plus_1_3y", "mtb_aa_plus_y3", "cdb_y3"),
        ("aaa_3_5y", "mtb_aaa_y5", "cdb_y5"),
        ("aa_1_3y", "mtb_aa_y3", "cdb_y3"),
    ]:
        if credit_col in wide.columns and base_col in wide.columns:
            wide[f"credit_spread_{bucket}"] = wide[credit_col] - wide[base_col]
        else:
            wide[f"credit_spread_{bucket}"] = np.nan
    wide["source"] = "wind_chinabond_curve_chunks_plus_existing_akshare"
    wide.to_parquet(PROCESSED / "credit_spreads_strict_daily_v42.parquet", index=False)
    return wide


def build_lgfv_province_spread() -> pd.DataFrame:
    cache = RAW_WIND / "analytics_credit_spreads_20180101_20260603.json"
    if not cache.exists():
        return pd.DataFrame()
    raw = blocks_to_df(json.loads(cache.read_text(encoding="utf-8")))
    if raw.empty:
        return pd.DataFrame()
    date = coalesce_date(raw)
    rows: list[pd.DataFrame] = []
    for c in raw.columns:
        s = str(c)
        if "城投利差" not in s:
            continue
        parts = s.split(":")
        province = parts[0] if parts else None
        rating = parts[-1] if parts else None
        vals = pd.to_numeric(raw[c], errors="coerce")
        tmp = pd.DataFrame({"date": date, "province": province, "rating": rating, "lgfv_spread_bp": vals})
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    out = out.dropna(subset=["date", "lgfv_spread_bp"])
    out["source"] = "wind_lgfv_spread_cache"
    out.to_parquet(PROCESSED / "lgfv_province_spreads_daily_v42.parquet", index=False)
    breadth = out.groupby("date", as_index=False).agg(
        lgfv_spread_province_count=("province", "nunique"),
        lgfv_spread_median=("lgfv_spread_bp", "median"),
        lgfv_spread_p75=("lgfv_spread_bp", lambda x: x.quantile(0.75)),
    )
    breadth.to_parquet(PROCESSED / "lgfv_spread_breadth_daily_v42.parquet", index=False)
    return out


def build_sw_industry_breadth() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pro = ts.pro_api(ts.get_token())
    cls = pro.index_classify(level="L1", src="SW2021")
    cls.to_parquet(PROCESSED / "sw2021_l1_industry_list.parquet", index=False)
    logs: list[dict] = []
    frames: list[pd.DataFrame] = []
    for _, r in cls.iterrows():
        code = r["index_code"]
        try:
            df = pro.sw_daily(ts_code=code, start_date=START_DATE, end_date=END_DATE)
            if not df.empty:
                df["industry_name"] = r["industry_name"]
                frames.append(df)
            logs.append({"dataset": "sw_industry_daily", "code": code, "name": r["industry_name"], "status": "ok", "rows": len(df)})
        except Exception as exc:
            logs.append({"dataset": "sw_industry_daily", "code": code, "name": r["industry_name"], "status": f"failed:{type(exc).__name__}", "rows": 0})
        time.sleep(0.08)
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if raw.empty:
        return raw, pd.DataFrame(), pd.DataFrame(logs)
    raw["date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    raw["pct_chg"] = pd.to_numeric(raw["pct_change"], errors="coerce")
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.sort_values(["ts_code", "date"])
    raw["ret_5d"] = raw.groupby("ts_code")["close"].pct_change(5)
    raw["ret_20d"] = raw.groupby("ts_code")["close"].pct_change(20)
    raw.to_parquet(PROCESSED / "sw_industry_returns_daily_v42.parquet", index=False)
    breadth = raw.groupby("date", as_index=False).agg(
        sw_l1_count=("ts_code", "nunique"),
        sw_l1_positive_1d_ratio=("pct_chg", lambda x: (x > 0).mean()),
        sw_l1_positive_5d_ratio=("ret_5d", lambda x: (x > 0).mean()),
        sw_l1_positive_20d_ratio=("ret_20d", lambda x: (x > 0).mean()),
        sw_l1_median_1d_ret=("pct_chg", "median"),
    )
    breadth["source"] = "tushare_sw_daily_SW2021_L1"
    breadth.to_parquet(PROCESSED / "equity_industry_breadth_daily_v42.parquet", index=False)
    return raw, breadth, pd.DataFrame(logs)


def build_merged_v42_panel() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for name in [
        "funding_rates_daily_v42.parquet",
        "omo_policy_daily_v42.parquet",
        "credit_spreads_strict_daily_v42.parquet",
        "equity_industry_breadth_daily_v42.parquet",
        "lgfv_spread_breadth_daily_v42.parquet",
    ]:
        p = PROCESSED / name
        if p.exists():
            df = pd.read_parquet(p)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                source_name = name.replace(".parquet", "").replace("_v42", "")
                rename = {c: f"{source_name}_{c}" for c in df.columns if c == "source"}
                if rename:
                    df = df.rename(columns=rename)
                parts.append(df.drop_duplicates("date"))
    merged: pd.DataFrame | None = None
    for df in parts:
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged = merged.sort_values("date")
    merged.to_parquet(PROCESSED / "v42_supplemental_factors_daily.parquet", index=False)
    merged.to_csv(PROCESSED / "v42_supplemental_factors_daily.csv", index=False, encoding="utf-8-sig")
    return merged


def coverage_row(name: str, df: pd.DataFrame, freq: str, required_cols: list[str] | None = None, note: str = "") -> dict:
    if df.empty:
        return {"dataset": name, "freq": freq, "rows": 0, "start": None, "end": None, "non_null_core": 0, "status": "缺", "note": note}
    date_col = "date" if "date" in df.columns else ("month" if "month" in df.columns else None)
    start = end = None
    if date_col:
        d = pd.to_datetime(df[date_col], errors="coerce")
        start = d.min()
        end = d.max()
    required_cols = required_cols or [c for c in df.columns if c not in ["date", "month", "source"]]
    existing = [c for c in required_cols if c in df.columns]
    non_null = int(df[existing].notna().any(axis=1).sum()) if existing else 0
    status = "已补齐" if non_null > 0 else "缺"
    return {"dataset": name, "freq": freq, "rows": len(df), "start": start, "end": end, "non_null_core": non_null, "status": status, "note": note}


def write_report(datasets: dict[str, pd.DataFrame], logs: pd.DataFrame) -> pd.DataFrame:
    rows = [
        coverage_row("macro_factors_monthly_v42", datasets.get("macro", pd.DataFrame()), "月频", ["cpi_yoy", "ppi_yoy", "m2_yoy", "pmi"], "宏观为月频，effective_date_conservative 防前视"),
        coverage_row("funding_rates_daily_v42", datasets.get("funding", pd.DataFrame()), "日频", ["r007", "dr007", "r007_dr007_spread"], "资金面日频"),
        coverage_row("omo_policy_daily_v42", datasets.get("omo", pd.DataFrame()), "日频", ["omo_net_injection", "omo_7d_rate"], "若 7D 逆回购利率为空，说明 Wind NL 未稳定返回该字段"),
        coverage_row("credit_spreads_strict_daily_v42", datasets.get("spreads", pd.DataFrame()), "日频", ["credit_spread_aaa_1_3y", "credit_spread_aa_plus_1_3y", "credit_spread_aaa_3_5y", "credit_spread_aa_1_3y"], "严格口径=信用债收益率-同期限国开债"),
        coverage_row("lgfv_province_spreads_daily_v42", datasets.get("lgfv", pd.DataFrame()), "日频", ["lgfv_spread_bp"], "区域城投利差缓存，当前主要用于区域扩散/广度"),
        coverage_row("equity_industry_breadth_daily_v42", datasets.get("breadth", pd.DataFrame()), "日频", ["sw_l1_positive_1d_ratio", "sw_l1_positive_5d_ratio"], "申万一级行业广度"),
        coverage_row("v42_supplemental_factors_daily", datasets.get("merged", pd.DataFrame()), "日频", None, "补充因子总表"),
    ]
    report = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    xlsx = REPORTS / "v42_gap_fill_report.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        report.to_excel(writer, sheet_name="coverage", index=False)
        if not logs.empty:
            logs.to_excel(writer, sheet_name="fetch_logs", index=False)
        for key, df in datasets.items():
            if not df.empty:
                df.head(1000).to_excel(writer, sheet_name=key[:31], index=False)
    md = REPORTS / "v42_gap_fill_report.md"
    md.write_text(
        "# v4.2 数据补全报告\n\n"
        + report.to_markdown(index=False)
        + "\n\n数据来源：Tushare、本地可转债数据库缓存、万得 Wind 金融数据服务。\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    logs_all: list[pd.DataFrame] = []
    macro = build_macro_monthly()
    funding, logs = build_funding_daily()
    logs_all.append(logs)
    omo, logs = build_omo_daily()
    logs_all.append(logs)
    curve_raw, logs = fetch_curve_quarters()
    logs_all.append(logs)
    spreads = build_credit_curves_daily(curve_raw)
    lgfv = build_lgfv_province_spread()
    sw_raw, breadth, logs = build_sw_industry_breadth()
    logs_all.append(logs)
    merged = build_merged_v42_panel()
    logs = pd.concat(logs_all, ignore_index=True, sort=False) if logs_all else pd.DataFrame()
    logs.to_parquet(PROCESSED / "v42_gap_fill_fetch_logs.parquet", index=False)
    report = write_report(
        {
            "macro": macro,
            "funding": funding,
            "omo": omo,
            "spreads": spreads,
            "lgfv": lgfv,
            "breadth": breadth,
            "merged": merged,
        },
        logs,
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
