from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_WIND = ROOT / "data_raw" / "wind" / "v43_remaining"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")


QUERIES = {
    "rating_downgrade_monthly": [
        "请返回2018年1月至2026年5月中国境内信用债主体评级或债项评级下调次数月度数据，只要月频，不要年度数据。字段包含月份、评级下调次数。",
        "请返回2018年1月至2026年5月债券评级下调月度统计，范围为中国信用债市场，只要月频数据，字段包含数据月份、下调主体数量、下调债项数量。",
        "请查询Wind中债券市场评级下调统计月度数据，2018年1月至2026年5月，返回月份和评级下调数量。",
    ],
    "industrial_profit_monthly": [
        "请返回2018年1月至2026年5月中国规模以上工业企业利润总额当月同比月度数据，只要月频，不要年度数据。字段包含月份、工业企业利润总额当月同比。",
        "请返回2018年1月至2026年5月中国工业企业利润总额累计同比月度数据，只要月频，不要年度数据。字段包含月份、工业企业利润累计同比。",
        "请查询国家统计局规模以上工业企业利润总额同比月度数据，2018年1月至2026年5月，只返回月频。",
    ],
    "lgfv_province_spreads": [
        "请返回2018年至2026年各省城投债信用利差日频或月频数据，字段包含日期、省份、城投债利差，不要年度数据。",
        "请查询Wind城投债分省信用利差，2018年1月至2026年5月，优先日频，没有日频则月频，返回日期、省份、利差。",
        "请返回中国各省城投债AAA或AA+信用利差月度数据，2018年1月至2026年5月，字段包含月份、省份、利差。",
    ],
    "reits_distribution_yield": [
        "请返回2018年至2026年中国公募REITs指数或中证REITs分派率、分红收益率、估值收益率月度数据，字段包含日期、分派率或分红收益率。",
        "请查询中国公募REITs分派率月度数据，2021年至2026年5月，返回月份、分派率。",
        "请返回中证REITs指数收盘价和分派率或股息率，2021年至2026年5月，优先日频，没有日频则月频。",
    ],
}


def key_pool() -> list[str | None]:
    raw = os.environ.get("WIND_API_KEY_POOL", "")
    keys = [x.strip() for x in raw.split(";") if x.strip()]
    return keys + [None]


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
            last_error = {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "stdout": res.stdout[:600], "stderr": res.stderr[:400]}
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


def blocks_to_df(data: object, dataset: str, query_id: int) -> pd.DataFrame:
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
        cols = [str(c.get("name", f"col_{i}")) for i, c in enumerate(block.get("columns") or [])]
        max_len = max(len(r) for r in rows)
        cols = (cols + [f"col_{i}" for i in range(len(cols), max_len)])[:max_len]
        seen: dict[str, int] = {}
        final_cols: list[str] = []
        for col in cols:
            n = seen.get(col, 0)
            final_cols.append(col if n == 0 else f"{col}__{n+1}")
            seen[col] = n + 1
        df = pd.DataFrame([list(r)[:max_len] + [None] * (max_len - len(r)) for r in rows], columns=final_cols)
        df["dataset"] = dataset
        df["query_id"] = query_id
        df["resolved_question"] = block.get("resolved_question", "")
        df["excel_total_count"] = block.get("excelTotalCount")
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def first_date(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in df.columns:
        s = str(col).lower()
        if str(col) in {"年份", "月份"}:
            continue
        if "日期" in str(col) or "时间" in str(col) or "month" in s or "date" in s:
            vals = pd.to_datetime(df[col], errors="coerce")
            out = out.combine_first(vals)
    if {"年份", "月份"}.issubset(df.columns):
        y = pd.to_numeric(df["年份"], errors="coerce")
        m = pd.to_numeric(df["月份"], errors="coerce")
        ym = pd.to_datetime(
            y.astype("Int64").astype(str) + "-" + m.astype("Int64").astype(str) + "-01",
            errors="coerce",
        ) + pd.offsets.MonthEnd(0)
        out = out.combine_first(ym)
    return out


def numeric_by_keywords(df: pd.DataFrame, keywords: list[str], excludes: list[str] | None = None) -> pd.Series:
    excludes = excludes or []
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for col in df.columns:
        name = str(col)
        if any(x in name for x in excludes):
            continue
        if all(k in name for k in keywords):
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                out = out.combine_first(vals.astype("Float64"))
    return out


def text_by_keywords(df: pd.DataFrame, keywords: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in df.columns:
        name = str(col)
        if all(k in name for k in keywords):
            vals = df[col].astype("object")
            if vals.notna().any():
                out = out.combine_first(vals)
    return out


def infer_freq(dates: pd.Series) -> str:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(d) < 3:
        return "unknown"
    med = d.diff().dt.days.dropna().median()
    if med <= 7:
        return "daily_or_trading_daily"
    if 25 <= med <= 35:
        return "monthly"
    if med >= 300:
        return "annual"
    return f"irregular_{med}"


def save_rating(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame({"month": first_date(df) + pd.offsets.MonthEnd(0)})
    out["rating_downgrade_count"] = numeric_by_keywords(df, ["下调"])
    out["rating_downgrade_subject_count"] = numeric_by_keywords(df, ["主体", "下调"])
    out["rating_downgrade_bond_count"] = numeric_by_keywords(df, ["债项", "下调"])
    out = out.dropna(subset=["month"]).drop_duplicates("month").sort_values("month")
    if infer_freq(out["month"]) == "annual" or out["rating_downgrade_count"].notna().sum() < 12:
        return pd.DataFrame()
    out["effective_date"] = out["month"] + pd.offsets.Day(15)
    out["source"] = "wind_analytics_rating_downgrade_monthly"
    out["quality_flag"] = "direct_monthly"
    out["frequency"] = "monthly"
    out["is_proxy"] = False
    out.to_parquet(PROCESSED / "rating_downgrade_monthly_v43.parquet", index=False)
    return out


def save_industrial_profit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dates = first_date(df) + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"month": dates})
    # Prefer official monthly YoY rows with an explicit date; fall back to cumulative YoY only when monthly YoY is unavailable.
    out["industrial_profit_yoy"] = numeric_by_keywords(df, ["利润总额", "当月同比"])
    out["industrial_profit_accum_yoy"] = numeric_by_keywords(df, ["利润总额", "累计同比"])
    out["industrial_profit_yoy"] = out["industrial_profit_yoy"].combine_first(out["industrial_profit_accum_yoy"])
    out = out.dropna(subset=["month"])
    out = out[out["month"].between(pd.Timestamp("2018-01-31"), pd.Timestamp("2026-05-31"))]
    out = out.sort_values(["month", "industrial_profit_yoy"]).drop_duplicates("month", keep="first").sort_values("month")
    if infer_freq(out["month"]) == "annual" or out["industrial_profit_yoy"].notna().sum() < 12:
        return pd.DataFrame()
    out["effective_date"] = out["month"] + pd.offsets.Day(27)
    out["source"] = "wind_analytics_industrial_profit_monthly"
    out["quality_flag"] = "direct_monthly"
    out["frequency"] = "monthly"
    out["is_proxy"] = False
    out.to_parquet(PROCESSED / "industrial_profit_monthly_v43.parquet", index=False)
    return out


def save_lgfv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame({"date": first_date(df)})
    province = text_by_keywords(df, ["省"]).combine_first(text_by_keywords(df, ["地区"]))
    out["province"] = province
    out["lgfv_spread"] = numeric_by_keywords(df, ["利差"])
    out = out.dropna(subset=["date", "lgfv_spread"]).drop_duplicates(["date", "province"]).sort_values(["date", "province"])
    freq = infer_freq(out["date"])
    if freq == "annual" or out["lgfv_spread"].notna().sum() < 12:
        return pd.DataFrame()
    out["source"] = "wind_analytics_lgfv_province_spreads"
    out["quality_flag"] = "direct_" + ("monthly" if freq == "monthly" else "daily")
    out["frequency"] = "monthly" if freq == "monthly" else "daily"
    out["is_proxy"] = False
    out.to_parquet(PROCESSED / "lgfv_province_spreads_v43.parquet", index=False)
    return out


def save_reits(df: pd.DataFrame) -> pd.DataFrame:
    deep = ROOT / "data_raw" / "wind" / "v43_deep_remaining" / "reits_index_yield_2.parquet"
    if deep.exists():
        raw = pd.read_parquet(deep)
        if {"年份", "月份"}.issubset(raw.columns):
            y = pd.to_numeric(raw["年份"], errors="coerce")
            m = pd.to_numeric(raw["月份"], errors="coerce")
            out = pd.DataFrame(
                {
                    "date": pd.to_datetime(y.astype("Int64").astype(str) + "-" + m.astype("Int64").astype(str) + "-01", errors="coerce")
                    + pd.offsets.MonthEnd(0),
                    "reits_distribution_yield": numeric_by_keywords(raw, ["平均", "分派率"]),
                    "reits_distribution_yield_median": numeric_by_keywords(raw, ["中位数", "分派率"]),
                    "reits_distribution_yield_count": numeric_by_keywords(raw, ["样本数量"]),
                }
            )
            out = out.dropna(subset=["date"])
            out = out[out["reits_distribution_yield"].notna() | out["reits_distribution_yield_median"].notna()].copy()
            if not out.empty:
                out["source"] = "wind_analytics_reits_market_average_distribution_yield"
                out["quality_flag"] = "direct_monthly"
                out["frequency"] = "monthly"
                out["is_proxy"] = False
                out["note"] = "Wind返回公募REITs市场整体月度平均/中位数分派率。"
                out.to_parquet(PROCESSED / "reits_distribution_yield_v43.parquet", index=False)
                return out
    if df.empty:
        return df
    out = pd.DataFrame({"date": first_date(df)})
    out["reits_distribution_yield"] = (
        numeric_by_keywords(df, ["分派率"])
        .combine_first(numeric_by_keywords(df, ["分红", "收益率"]))
        .combine_first(numeric_by_keywords(df, ["股息率"]))
    )
    out["reits_index_close"] = numeric_by_keywords(df, ["收盘"])
    out["wind_code"] = df["Wind代码"] if "Wind代码" in df.columns else pd.NA
    out["security_name"] = df["证券简称"] if "证券简称" in df.columns else pd.NA
    out = out.dropna(subset=["date"]).sort_values("date")
    freq = infer_freq(out["date"])
    if freq == "annual" or out["reits_distribution_yield"].notna().sum() < 3:
        return pd.DataFrame()
    # Wind returned single-product REIT rows. Aggregate to month-end market-average distribution yield.
    out["month"] = out["date"] + pd.offsets.MonthEnd(0)
    agg = out.groupby("month", as_index=False).agg(
        reits_distribution_yield=("reits_distribution_yield", "mean"),
        reits_distribution_yield_median=("reits_distribution_yield", "median"),
        reits_distribution_yield_count=("reits_distribution_yield", "count"),
        reits_index_close=("reits_index_close", "mean"),
    )
    agg = agg[agg["reits_distribution_yield_count"] > 0].copy()
    agg = agg.rename(columns={"month": "date"})
    agg["source"] = "wind_analytics_reits_single_product_distribution_yield_monthly_average"
    agg["quality_flag"] = "proxy_monthly"
    agg["frequency"] = "monthly"
    agg["is_proxy"] = True
    agg["note"] = "Wind未返回指数分派率，使用单只公募REITs分派率的月度均值/中位数代理。"
    agg.to_parquet(PROCESSED / "reits_distribution_yield_v43.parquet", index=False)
    out.to_parquet(PROCESSED / "reits_distribution_yield_single_product_v43.parquet", index=False)
    return agg


def main() -> None:
    RAW_WIND.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    fetch_rows = []
    parsed: dict[str, pd.DataFrame] = {}
    for dataset, questions in QUERIES.items():
        frames = []
        for i, q in enumerate(questions, start=1):
            cache = RAW_WIND / f"{dataset}_{i}.json"
            data, status = call_wind(q, cache)
            if status.startswith("ok"):
                df = blocks_to_df(data, dataset, i)
                if not df.empty:
                    frames.append(df)
                fetch_rows.append({"dataset": dataset, "query_id": i, "status": status, "rows": len(df), "cache": str(cache), "question": q})
            else:
                (RAW_WIND / f"{dataset}_{i}_error.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                fetch_rows.append({"dataset": dataset, "query_id": i, "status": status, "rows": 0, "cache": str(cache), "question": q, "error": json.dumps(data, ensure_ascii=False)[:500]})
        parsed[dataset] = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if not parsed[dataset].empty:
            parsed[dataset].to_csv(RAW_WIND / f"{dataset}_parsed.csv", index=False, encoding="utf-8-sig")
            parsed[dataset].to_parquet(RAW_WIND / f"{dataset}_parsed.parquet", index=False)

    outputs = {
        "rating_downgrade_monthly_v43": save_rating(parsed["rating_downgrade_monthly"]),
        "industrial_profit_monthly_v43": save_industrial_profit(parsed["industrial_profit_monthly"]),
        "lgfv_province_spreads_v43": save_lgfv(parsed["lgfv_province_spreads"]),
        "reits_distribution_yield_v43": save_reits(parsed["reits_distribution_yield"]),
    }
    report_rows = []
    for name, df in outputs.items():
        dcol = "date" if "date" in df.columns else "month" if "month" in df.columns else None
        dates = pd.to_datetime(df[dcol], errors="coerce") if dcol and not df.empty else pd.Series(dtype="datetime64[ns]")
        report_rows.append(
            {
                "dataset": name,
                "saved": not df.empty,
                "rows": len(df),
                "start": dates.min() if len(dates) else None,
                "end": dates.max() if len(dates) else None,
                "frequency": df["frequency"].iloc[0] if not df.empty and "frequency" in df.columns else None,
                "quality_flag": df["quality_flag"].iloc[0] if not df.empty and "quality_flag" in df.columns else None,
                "note": "annual/empty/insufficient data rejected" if df.empty else "",
            }
        )
    fetch_report = pd.DataFrame(fetch_rows)
    output_report = pd.DataFrame(report_rows)
    fetch_report.to_csv(REPORTS / "v43_remaining_wind_fetch_log.csv", index=False, encoding="utf-8-sig")
    output_report.to_csv(REPORTS / "v43_remaining_wind_outputs.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(REPORTS / "v43_remaining_wind_fetch_report.xlsx") as writer:
        fetch_report.to_excel(writer, sheet_name="fetch_log", index=False)
        output_report.to_excel(writer, sheet_name="outputs", index=False)
        for dataset, df in parsed.items():
            if not df.empty:
                df.head(1000).to_excel(writer, sheet_name=dataset[:31], index=False)
    print(output_report.to_string(index=False))


if __name__ == "__main__":
    main()
