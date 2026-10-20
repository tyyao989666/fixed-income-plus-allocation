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
    keys = [x.strip() for x in raw.split(";") if x.strip()]
    return keys + [None]


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
        cmd = [
            "node",
            str(SKILL_CLI),
            "call",
            server,
            tool,
            json.dumps(params, ensure_ascii=False, separators=(",", ":")),
        ]
        res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=180, env=env)
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


def month_end(df: pd.DataFrame, date_col: str, value_map: dict[str, str], prefix_cols: dict[str, str] | None = None) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp["date"].notna()].sort_values("date")
    if tmp.empty:
        return pd.DataFrame()
    rows = []
    for m, g in tmp.groupby(tmp["date"].dt.to_period("M")):
        last = g.iloc[-1]
        row = {"month": m.to_timestamp("M")}
        if prefix_cols:
            for out_col, src_col in prefix_cols.items():
                row[out_col] = last.get(src_col)
        for out_col, src_col in value_map.items():
            row[out_col] = pd.to_numeric(last.get(src_col), errors="coerce")
        rows.append(row)
    return pd.DataFrame(rows)


def chunks_by_quarter() -> list[tuple[str, str, str]]:
    out = []
    periods = pd.period_range("2018Q1", "2026Q2", freq="Q")
    for p in periods:
        start = p.start_time.strftime("%Y%m%d")
        end = min(p.end_time, pd.Timestamp("2026-06-04")).strftime("%Y%m%d")
        out.append((p.strftime("%YQ%q"), start, end))
    return out


def chunks_by_half_year() -> list[tuple[str, str, str]]:
    out = []
    for year in range(2018, 2027):
        for h, start, end in [
            ("H1", f"{year}0101", f"{year}0630"),
            ("H2", f"{year}0701", f"{year}1231"),
        ]:
            if start > END:
                continue
            out.append((f"{year}{h}", start, min(end, END)))
    return out


def build_index_valuation() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = {
        "csi1000": ("中证1000", "000852.SH"),
        "dividend": ("中证红利", "000922.CSI"),
        "csi_convertible_bond": ("中证转债", "000832.CSI"),
    }
    all_monthly = []
    logs = []
    for name, (cn, code) in targets.items():
        raw_frames = []
        for tag, start, end in chunks_by_quarter():
            q = f"查询{start}至{end}{cn}指数{code}每日PE、PB、收盘价、成交额，返回日期、PE、PB、收盘价、成交额。"
            data, status = call_wind(
                "index_data",
                "get_index_fundamentals",
                {"question": q, "lang": "中文"},
                RAW / "index_valuation" / f"{name}_{tag}.json",
            )
            df = blocks_to_df(data)
            if not df.empty:
                df["index_name"] = name
                df["wind_code"] = code
                raw_frames.append(df)
            logs.append({"dataset": "index_valuation", "name": name, "chunk": tag, "status": status, "rows": len(df)})
        raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
        if not raw.empty:
            date_col = next((c for c in raw.columns if "日期" in c), None)
            pe_col = next((c for c in raw.columns if "PE" in c.upper()), None)
            pb_col = next((c for c in raw.columns if "PB" in c.upper()), None)
            close_col = next((c for c in raw.columns if "收盘" in c or "MATCH" in c), None)
            amt_col = next((c for c in raw.columns if "成交额" in c), None)
            mapping = {}
            if pe_col:
                mapping[f"{name}_pe"] = pe_col
            if pb_col:
                mapping[f"{name}_pb"] = pb_col
            if close_col:
                mapping[f"{name}_close"] = close_col
            if amt_col:
                mapping[f"{name}_amount"] = amt_col
            m = month_end(raw, date_col, mapping)
            if not m.empty:
                m["source"] = "wind_index_fundamentals_quarterly_chunks"
                all_monthly.append(m)
        raw.to_parquet(PROCESSED / f"raw_wind_{name}_valuation_daily.parquet", index=False)
    merged = None
    for m in all_monthly:
        merged = m if merged is None else merged.merge(m.drop(columns=["source"], errors="ignore"), on="month", how="outer")
    out = merged.sort_values("month") if merged is not None else pd.DataFrame()
    if not out.empty and "source" not in out.columns:
        out["source"] = "wind_index_fundamentals_quarterly_chunks"
    return out, pd.DataFrame(logs)


def build_alt_overseas() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = {
        "us_10y_yield": "美国10年期国债收益率",
        "usd_index": "美元指数收盘价",
        "nanhua_commodity": "南华商品指数收盘价",
        "gold_au9999": "上海黄金交易所Au9999收盘价",
    }
    questions = {
        "us_10y_yield": "查询{start}至{end}美国10年期国债收益率日频数据，返回日期和值。",
        "usd_index": "查询{start}至{end}美元指数日频收盘价，返回日期和值。",
        "nanhua_commodity": "查询{start}至{end}南华商品指数日频收盘价，返回日期和值。",
        "gold_au9999": "查询{start}至{end}上海黄金交易所Au9999日频收盘价，返回日期和值。",
    }
    monthly_frames = []
    logs = []
    for name in targets:
        raw_frames = []
        for tag, start, end in chunks_by_half_year():
            q = questions[name].format(start=start, end=end)
            data, status = call_wind(
                "analytics_data",
                "get_financial_data",
                {"question": q, "lang": "中文"},
                RAW / "alt_overseas" / f"{name}_{tag}.json",
            )
            df = blocks_to_df(data)
            if not df.empty:
                df["factor"] = name
                raw_frames.append(df)
            logs.append({"dataset": "alt_overseas", "name": name, "chunk": tag, "status": status, "rows": len(df)})
        raw = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
        raw.to_parquet(PROCESSED / f"raw_wind_{name}_daily.parquet", index=False)
        if raw.empty:
            continue
        date_col = next((c for c in raw.columns if "日期" in c or "交易时间" in c), None)
        candidates = []
        for c in raw.columns:
            s = str(c)
            if c in {"factor", "resolved_question"} or "日期" in s or "代码" in s or "名称" in s or "币种" in s:
                continue
            if pd.to_numeric(raw[c], errors="coerce").notna().any():
                candidates.append(c)
        value_col = candidates[-1] if candidates else None
        if date_col and value_col:
            m = month_end(raw, date_col, {name: value_col})
            monthly_frames.append(m)
    merged = None
    for m in monthly_frames:
        merged = m if merged is None else merged.merge(m, on="month", how="outer")
    out = merged.sort_values("month") if merged is not None else pd.DataFrame()
    if not out.empty:
        for c in [x for x in out.columns if x != "month"]:
            out[f"{c}_ret_1m"] = out[c].pct_change()
        out["source"] = "wind_analytics_half_year_chunks"
    return out, pd.DataFrame(logs)


def build_aaa_cd_proxy() -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = []
    frames = []
    # ChinaBond public curve proxy: commercial bank ordinary bond AAA 1Y.
    for year in range(2018, 2027):
        try:
            df = ak.bond_china_yield(start_date=f"{year}0101", end_date=min(f"{year}1231", END))
            logs.append({"dataset": "aaa_cd_proxy", "source": "akshare_chinabond", "year": year, "status": "ok", "rows": len(df)})
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logs.append({"dataset": "aaa_cd_proxy", "source": "akshare_chinabond", "year": year, "status": f"failed:{exc}", "rows": 0})
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    raw.to_parquet(PROCESSED / "raw_chinabond_yield_curves.parquet", index=False)
    bank = raw[raw.get("曲线名称", "").astype(str).str.contains("商业银行普通债收益率曲线\\(AAA\\)", regex=True)].copy() if not raw.empty else pd.DataFrame()
    out = pd.DataFrame()
    if not bank.empty:
        bank["date"] = pd.to_datetime(bank["日期"], errors="coerce")
        bank = bank.dropna(subset=["date"]).sort_values("date")
        rows = []
        for m, g in bank.groupby(bank["date"].dt.to_period("M")):
            last = g.iloc[-1]
            rows.append({"month": m.to_timestamp("M"), "aaa_cd_1y_proxy": pd.to_numeric(last.get("1年"), errors="coerce")})
        out = pd.DataFrame(rows)
    shibor_path = ROOT / "data_raw" / "tushare" / "shibor_shibor_20180101_20260603.parquet"
    if shibor_path.exists():
        sh = pd.read_parquet(shibor_path)
        sh["date"] = pd.to_datetime(sh["date"], errors="coerce")
        sh = sh.dropna(subset=["date"]).sort_values("date")
        rows = []
        for m, g in sh.groupby(sh["date"].dt.to_period("M")):
            last = g.iloc[-1]
            rows.append({"month": m.to_timestamp("M"), "shibor_3m_proxy": pd.to_numeric(last.get("3m"), errors="coerce"), "shibor_1y_proxy": pd.to_numeric(last.get("1y"), errors="coerce")})
        sh_m = pd.DataFrame(rows)
        out = sh_m if out.empty else out.merge(sh_m, on="month", how="outer")
    if not out.empty:
        out["aaa_cd_1y_final"] = out["aaa_cd_1y_proxy"].combine_first(out.get("shibor_1y_proxy"))
        out["source"] = "proxy:chinabond_commercial_bank_aaa_1y; fallback:tushare_shibor_1y"
        out["note"] = "未取得中债商业银行同业存单AAA:1年完整真值；此表为状态引擎代理口径。"
    return out.sort_values("month") if not out.empty else out, pd.DataFrame(logs)


def build_rating_proxy() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Probe is intentionally documented; no reliable table was returned by Wind NL.
    probe_q = "统计2018年1月至2026年6月信用债主体评级下调次数和债项评级下调次数，按月份汇总，返回月份、主体评级下调次数、债项评级下调次数。"
    data, status = call_wind(
        "analytics_data",
        "get_financial_data",
        {"question": probe_q, "lang": "中文"},
        RAW / "rating_downgrade" / "rating_downgrade_monthly_probe.json",
    )
    probe_df = blocks_to_df(data)
    logs = [{"dataset": "rating_downgrade", "status": status, "rows": len(probe_df), "note": "Wind NL did not return reliable monthly downgrade table" if probe_df.empty else ""}]

    tail_path = PROCESSED / "wind_credit_tail_monthly.parquet"
    proxy = pd.DataFrame()
    if tail_path.exists():
        tail = pd.read_parquet(tail_path)
        date_col = "month" if "month" in tail.columns else "date"
        tail["month"] = pd.to_datetime(tail[date_col], errors="coerce")
        nums = [c for c in ["cancel_issuance_count", "cancel_issuance_amount", "credit_default_count", "credit_default_amount"] if c in tail.columns]
        proxy = tail[["month"] + nums].dropna(subset=["month"]).copy()
        for c in nums:
            proxy[c] = pd.to_numeric(proxy[c], errors="coerce").fillna(0)
        if nums:
            base = sum(proxy[c] for c in nums)
            rolling = base.rolling(24, min_periods=6)
            proxy["rating_downgrade_pressure_proxy"] = (base - rolling.mean()) / rolling.std().replace(0, pd.NA)
        proxy["actual_rating_downgrade_count"] = pd.NA
        proxy["source"] = "proxy_from_cancel_issuance_and_default; actual_rating_downgrade_missing"
        proxy["note"] = "未取得可靠月频评级下调真值，禁止当作真实评级下调次数使用。"
    return proxy.sort_values("month") if not proxy.empty else proxy, pd.DataFrame(logs)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    ix, ix_log = build_index_valuation()
    alt, alt_log = build_alt_overseas()
    cd, cd_log = build_aaa_cd_proxy()
    rating, rating_log = build_rating_proxy()

    ix.to_parquet(PROCESSED / "index_valuation_filled_monthly.parquet", index=False)
    alt.to_parquet(PROCESSED / "alt_overseas_filled_monthly.parquet", index=False)
    cd.to_parquet(PROCESSED / "aaa_cd_1y_proxy_monthly.parquet", index=False)
    rating.to_parquet(PROCESSED / "rating_downgrade_proxy_monthly.parquet", index=False)

    logs = pd.concat([ix_log, alt_log, cd_log, rating_log], ignore_index=True, sort=False)
    logs.to_csv(REPORTS / "hard_gap_fill_log.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(REPORTS / "hard_gap_fill_report.xlsx", engine="openpyxl") as writer:
        ix.to_excel(writer, sheet_name="index_valuation", index=False)
        alt.to_excel(writer, sheet_name="alt_overseas", index=False)
        cd.to_excel(writer, sheet_name="aaa_cd_proxy", index=False)
        rating.to_excel(writer, sheet_name="rating_proxy", index=False)
        logs.to_excel(writer, sheet_name="fetch_log", index=False)
    lines = [
        "# Hard Gap Fill Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Outputs",
        "- index_valuation_filled_monthly.parquet: 中证1000/红利/中证转债估值或成交额补表。",
        "- alt_overseas_filled_monthly.parquet: 美债10Y、美元指数、南华商品、黄金月频补表。",
        "- aaa_cd_1y_proxy_monthly.parquet: AAA同业存单1Y代理表。",
        "- rating_downgrade_proxy_monthly.parquet: 评级下调缺口代理表，非真实评级下调次数。",
        "",
        "## Coverage",
        f"- index valuation rows: {len(ix)}",
        f"- alt overseas rows: {len(alt)}",
        f"- AAA CD proxy rows: {len(cd)}",
        f"- rating proxy rows: {len(rating)}",
        "",
        "数据来源于万得 Wind 金融数据服务；代理表同时使用 AkShare/中债公开曲线及 Tushare Shibor。",
    ]
    (REPORTS / "hard_gap_fill_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame([
        {"table": "index_valuation_filled_monthly", "rows": len(ix), "start": ix["month"].min() if "month" in ix else None, "end": ix["month"].max() if "month" in ix else None},
        {"table": "alt_overseas_filled_monthly", "rows": len(alt), "start": alt["month"].min() if "month" in alt else None, "end": alt["month"].max() if "month" in alt else None},
        {"table": "aaa_cd_1y_proxy_monthly", "rows": len(cd), "start": cd["month"].min() if "month" in cd else None, "end": cd["month"].max() if "month" in cd else None},
        {"table": "rating_downgrade_proxy_monthly", "rows": len(rating), "start": rating["month"].min() if "month" in rating else None, "end": rating["month"].max() if "month" in rating else None},
    ]).to_string(index=False))


if __name__ == "__main__":
    main()
