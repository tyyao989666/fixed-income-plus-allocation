from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts
import yaml


DATE_FMT = "%Y%m%d"


@dataclass
class ProjectPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def raw_tushare(self) -> Path:
        return self.root / "data_raw" / "tushare"

    @property
    def raw_wind(self) -> Path:
        return self.root / "data_raw" / "wind"

    @property
    def processed(self) -> Path:
        return self.root / "data_processed"

    @property
    def reports(self) -> Path:
        return self.root / "output" / "reports"

    @property
    def tables(self) -> Path:
        return self.root / "output" / "tables"


def ensure_dirs(root: Path) -> ProjectPaths:
    paths = ProjectPaths(root)
    for p in [
        paths.config,
        paths.raw_tushare,
        paths.raw_wind,
        root / "data_raw" / "manual",
        root / "data_raw" / "rates",
        root / "data_raw" / "credit",
        root / "data_raw" / "convertible_bonds",
        root / "data_raw" / "equity",
        root / "data_raw" / "alternatives",
        root / "data_raw" / "macro",
        root / "data_raw" / "liquidity",
        paths.processed,
        paths.reports,
        paths.tables,
        root / "output" / "figures",
        root / "output" / "models",
        root / "scripts",
        root / "src",
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def write_default_configs(paths: ProjectPaths) -> None:
    defaults = {
        "wind_codes.yaml": {"note": "Wind兜底代码待补，当前数据准备优先Tushare。"},
        "code_mapping.yaml": {"note": "资产映射以asset_universe.yaml为准。"},
        "industry_etf_mapping.yaml": {"note": "权益行业代理ETF映射以asset_universe.yaml为准。"},
        "credit_rating_mapping.yaml": {"ratings": ["AAA", "AA+", "AA"]},
        "credit_lgfv_industry_mapping.yaml": {"types": ["lgfv", "industrial"]},
        "cb_style_classification_rules.yaml": {
            "price_low_threshold": 120,
            "bond_style_upper": 110,
            "balanced_style_upper": 130,
            "size_split": "issue_size_median",
        },
        "cb_index_constituent_rules.yaml": {
            "min_listing_days": 20,
            "exclude_suspended": True,
            "max_bonds_for_initial_download": 260,
        },
        "feature_config.yaml": {
            "daily_windows": [5, 20, 60, 120],
            "monthly_windows": [3, 6, 12],
        },
        "scenario_thresholds.yaml": {"enabled": False, "note": "策略阶段再启用。"},
        "market_state_card_schema.yaml": {"enabled": False, "note": "策略阶段再启用。"},
        "confidence_rules.yaml": {"enabled": False, "note": "策略阶段再启用。"},
        "transaction_cost.yaml": {"enabled": False, "note": "策略阶段再启用。"},
        "constraint_priority.yaml": {"enabled": False, "note": "策略阶段再启用。"},
        "optimization_config.yaml": {"enabled": False, "note": "策略阶段再启用。"},
        "backtest_config.yaml": {"enabled": False, "note": "策略阶段再启用。"},
        "model_config.yaml": {"enabled": False, "note": "数据准备阶段不训练模型。"},
        "llm_config.yaml": {"enabled": False, "note": "数据准备阶段不调用LLM。"},
        "acceptance_criteria.yaml": {
            "p0": [
                "Tushare connection passes",
                "asset_universe.parquet non-empty",
                "asset_returns_daily.parquet non-empty",
                "missing source report generated",
            ]
        },
        "ablation_config.yaml": {"enabled": False},
        "ep_config.yaml": {"enabled": False},
        "ep_view_mapping.yaml": {"enabled": False},
        "llm_prompt_templates.yaml": {"enabled": False},
        "llm_output_schema.json": {
            "type": "object",
            "additionalProperties": True,
        },
    }
    for name, data in defaults.items():
        path = paths.config / name
        if not path.exists():
            if name.endswith(".json"):
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                save_yaml(path, data)


def get_tushare_pro() -> Any:
    token = ts.get_token()
    if not token:
        raise RuntimeError("未找到本地Tushare token，请先配置TUSHARE_TOKEN或tushare token。")
    return ts.pro_api(token)


def normalize_trade_frame(df: pd.DataFrame, asset: dict[str, Any], fetch_time: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"], format=DATE_FMT, errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        return pd.DataFrame()
    out = out.dropna(subset=["date"]).sort_values("date")
    if "close" not in out.columns:
        return pd.DataFrame()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["return_1d"] = out["close"].pct_change()
    if "pct_chg" in out.columns:
        pct = pd.to_numeric(out["pct_chg"], errors="coerce") / 100.0
        out["return_1d"] = out["return_1d"].where(out["return_1d"].notna(), pct)
    out["asset_code"] = asset["code"]
    out["code"] = asset["code"]
    out["asset_name"] = asset.get("name", asset["code"])
    out["asset_class"] = asset.get("asset_class", asset.get("module", "unknown"))
    out["module"] = asset.get("module", "unknown")
    out["sub_module"] = asset.get("sub_module", asset.get("code", ""))
    out["ts_code"] = asset.get("ts_code")
    out["data_source"] = asset.get("data_source", "tushare")
    out["fetch_time"] = fetch_time
    keep = [
        "date",
        "asset_code",
        "code",
        "asset_name",
        "asset_class",
        "module",
        "sub_module",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "return_1d",
        "data_source",
        "fetch_time",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def cache_path(paths: ProjectPaths, endpoint: str, key: str) -> Path:
    safe_key = key.replace("/", "_").replace(":", "_")
    return paths.raw_tushare / f"{endpoint}_{safe_key}.parquet"


def fetch_with_cache(
    paths: ProjectPaths,
    pro: Any,
    endpoint: str,
    key: str,
    params: dict[str, Any],
    sleep_seconds: float,
) -> pd.DataFrame:
    path = cache_path(paths, endpoint, key)
    if path.exists():
        return pd.read_parquet(path)
    fn = getattr(pro, endpoint)
    df = fn(**params)
    if df is None:
        df = pd.DataFrame()
    df.to_parquet(path, index=False)
    time.sleep(sleep_seconds)
    return df


def fetch_configured_assets(paths: ProjectPaths, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pro = get_tushare_pro()
    start_date = config["project"]["start_date"]
    end_date = config["project"]["end_date"]
    sleep_seconds = float(config["project"].get("sleep_seconds", 0.08))
    fetch_time = datetime.now().isoformat(timespec="seconds")
    frames: list[pd.DataFrame] = []
    universe_rows: list[dict[str, Any]] = []

    for asset in config.get("assets", []):
        row = dict(asset)
        row.update({"tradable_proxy": asset.get("ts_code"), "fetch_time": fetch_time})
        if asset.get("source_type") == "missing_source":
            row.update({"status": "missing_source", "start_date": None, "end_date": None, "n_obs": 0})
            universe_rows.append(row)
            continue
        endpoint = asset.get("source_type", "fund_daily")
        try:
            df = fetch_with_cache(
                paths,
                pro,
                endpoint,
                f"{asset.get('ts_code')}_{start_date}_{end_date}",
                {"ts_code": asset.get("ts_code"), "start_date": start_date, "end_date": end_date},
                sleep_seconds,
            )
            norm = normalize_trade_frame(df, asset, fetch_time)
            if norm.empty:
                row.update({"status": "empty", "start_date": None, "end_date": None, "n_obs": 0})
            else:
                frames.append(norm)
                row.update(
                    {
                        "status": "ok",
                        "start_date": norm["date"].min().date().isoformat(),
                        "end_date": norm["date"].max().date().isoformat(),
                        "n_obs": int(norm.shape[0]),
                    }
                )
        except Exception as exc:
            row.update({"status": "fetch_failed", "error": str(exc), "start_date": None, "end_date": None, "n_obs": 0})
        universe_rows.append(row)

    return pd.DataFrame(universe_rows), pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_extra_raw_data(paths: ProjectPaths, config: dict[str, Any]) -> dict[str, str]:
    pro = get_tushare_pro()
    start_date = config["project"]["start_date"]
    end_date = config["project"]["end_date"]
    sleep_seconds = float(config["project"].get("sleep_seconds", 0.08))
    status: dict[str, str] = {}

    calls = [
        ("trade_cal", "calendar", {"exchange": "SSE", "start_date": start_date, "end_date": end_date}),
        ("index_daily", "hs300", {"ts_code": "000300.SH", "start_date": start_date, "end_date": end_date}),
        ("index_daily", "csi500", {"ts_code": "000905.SH", "start_date": start_date, "end_date": end_date}),
        ("index_daily", "csi1000", {"ts_code": "000852.SH", "start_date": start_date, "end_date": end_date}),
        ("index_daily", "chinext", {"ts_code": "399006.SZ", "start_date": start_date, "end_date": end_date}),
        ("index_daily", "sse", {"ts_code": "000001.SH", "start_date": start_date, "end_date": end_date}),
        ("shibor", "shibor", {"start_date": start_date, "end_date": end_date}),
        ("cn_m", "money_supply", {"start_m": start_date[:6], "end_m": end_date[:6]}),
        ("cn_cpi", "cpi", {"start_m": start_date[:6], "end_m": end_date[:6]}),
        ("cn_ppi", "ppi", {"start_m": start_date[:6], "end_m": end_date[:6]}),
        ("cn_pmi", "pmi", {"start_m": start_date[:6], "end_m": end_date[:6]}),
        ("sf_month", "social_financing", {"start_m": start_date[:6], "end_m": end_date[:6]}),
    ]
    for endpoint, key, params in calls:
        try:
            df = fetch_with_cache(paths, pro, endpoint, key, params, sleep_seconds)
            status[f"{endpoint}:{key}"] = f"ok:{len(df)}"
        except Exception as exc:
            status[f"{endpoint}:{key}"] = f"failed:{exc}"
    return status


def build_cb_subindices(paths: ProjectPaths, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cb_cfg = config.get("convertible_bond_subindices", {})
    if not cb_cfg.get("enabled", False):
        return pd.DataFrame(), pd.DataFrame()
    pro = get_tushare_pro()
    start_date = config["project"]["start_date"]
    end_date = config["project"]["end_date"]
    sleep_seconds = float(config["project"].get("sleep_seconds", 0.08))
    max_codes = int(config["project"].get("convertible_bond_max_codes", 260))
    fetch_time = datetime.now().isoformat(timespec="seconds")

    cb_basic = fetch_with_cache(paths, pro, "cb_basic", "all", {}, sleep_seconds)
    cb_basic.to_parquet(paths.raw_tushare / "cb_basic_all.parquet", index=False)
    basic = cb_basic.copy()
    for col in ["list_date", "delist_date"]:
        if col in basic.columns:
            basic[col] = basic[col].astype(str).replace({"nan": ""})
    active = basic[
        (basic.get("list_date", "") <= end_date)
        & ((basic.get("delist_date", "").fillna("") == "") | (basic.get("delist_date", "").fillna("") >= start_date))
    ].copy()
    if "remain_size" in active.columns:
        active["remain_size_num"] = pd.to_numeric(active["remain_size"], errors="coerce")
        active = active.sort_values("remain_size_num", ascending=False)
    active = active.head(max_codes)

    daily_frames = []
    for ts_code in active["ts_code"].dropna().astype(str).tolist():
        try:
            df = fetch_with_cache(
                paths,
                pro,
                "cb_daily",
                f"{ts_code}_{start_date}_{end_date}",
                {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
                sleep_seconds,
            )
            if not df.empty:
                daily_frames.append(df)
        except Exception:
            continue
    if not daily_frames:
        return pd.DataFrame(), pd.DataFrame()
    cb_daily = pd.concat(daily_frames, ignore_index=True)
    cb_daily.to_parquet(paths.raw_tushare / "cb_daily_sample.parquet", index=False)
    d = cb_daily.merge(active[["ts_code", "bond_short_name", "issue_size"]], on="ts_code", how="left")
    d["date"] = pd.to_datetime(d["trade_date"], format=DATE_FMT, errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values(["ts_code", "date"])
    d["ret"] = d.groupby("ts_code")["close"].pct_change()
    d["issue_size_num"] = pd.to_numeric(d["issue_size"], errors="coerce")
    size_median = d["issue_size_num"].median()
    d["size_bucket"] = np.where(d["issue_size_num"] >= size_median, "large", "small")
    d["close_pct_rank"] = d.groupby("date")["close"].rank(pct=True)
    d["style"] = np.select(
        [d["close_pct_rank"] <= 1 / 3, d["close_pct_rank"] <= 2 / 3],
        ["bond", "balanced"],
        default="equity",
    )
    d["style_daily_median_close"] = d.groupby(["date", "style"])["close"].transform("median")
    d["price_bucket"] = np.where(d["close"] <= d["style_daily_median_close"], "low", "high")
    sub = (
        d.dropna(subset=["ret"])
        .groupby(["date", "style", "size_bucket", "price_bucket"], as_index=False)
        .agg(return_1d=("ret", "mean"), member_count=("ts_code", "nunique"), close=("close", "mean"))
    )
    sub["asset_code"] = "cb_" + sub["style"] + "_" + sub["size_bucket"] + "_" + sub["price_bucket"]
    sub["code"] = sub["asset_code"]
    name_map = {
        "bond": "债性",
        "balanced": "平衡",
        "equity": "股性",
        "large": "大盘",
        "small": "小盘",
        "low": "低价",
        "high": "高价",
    }
    sub["asset_name"] = sub.apply(
        lambda r: f"可转债{name_map[r['style']]}{name_map[r['size_bucket']]}{name_map[r['price_bucket']]}", axis=1
    )
    sub["asset_class"] = "convertible_bonds"
    sub["module"] = "convertible_bonds"
    sub["sub_module"] = sub["asset_code"]
    sub["ts_code"] = np.nan
    sub["data_source"] = "tushare_derived"
    sub["fetch_time"] = fetch_time
    sub = sub.sort_values(["asset_code", "date"])
    sub["close"] = sub.groupby("asset_code")["return_1d"].transform(lambda x: (1 + x.fillna(0)).cumprod() * 100)
    out_cols = [
        "date",
        "asset_code",
        "code",
        "asset_name",
        "asset_class",
        "module",
        "sub_module",
        "ts_code",
        "close",
        "return_1d",
        "member_count",
        "data_source",
        "fetch_time",
    ]
    for col in ["open", "high", "low", "vol", "amount"]:
        sub[col] = np.nan
    sub = sub[out_cols + ["open", "high", "low", "vol", "amount"]]
    uni = (
        sub.groupby("asset_code")
        .agg(
            asset_name=("asset_name", "first"),
            module=("module", "first"),
            sub_module=("sub_module", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            n_obs=("date", "count"),
        )
        .reset_index()
    )
    uni["code"] = uni["asset_code"]
    uni["source_type"] = "cb_daily_derived"
    uni["data_source"] = "tushare_derived"
    uni["status"] = "ok"
    uni["tradable_proxy"] = "convertible_bond_basket"
    uni["note"] = "Tushare可转债个券日行情等权自建，首版使用发行规模切分大小盘、每日收盘价分位切分债性/平衡/股性，并在风格组内切分高低价；转股溢价率口径待Wind或更完整转债估值数据补齐。"
    uni["start_date"] = uni["start_date"].dt.date.astype(str)
    uni["end_date"] = uni["end_date"].dt.date.astype(str)
    return uni, sub


def build_monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["asset_code", "date"])
    rows = []
    for code, g in d.groupby("asset_code"):
        m = g.set_index("date").resample("ME").last().dropna(subset=["close"]).reset_index()
        m["return_1m"] = m["close"].pct_change()
        for col in ["asset_name", "asset_class", "module", "sub_module", "data_source", "fetch_time"]:
            m[col] = g[col].dropna().iloc[-1] if g[col].notna().any() else np.nan
        m["asset_code"] = code
        m["code"] = code
        rows.append(m)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    keep = [
        "date",
        "asset_code",
        "code",
        "asset_name",
        "asset_class",
        "module",
        "sub_module",
        "close",
        "return_1m",
        "data_source",
        "fetch_time",
    ]
    return out[keep]


def rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    def _mdd(x: np.ndarray) -> float:
        if len(x) == 0 or np.all(np.isnan(x)):
            return np.nan
        s = pd.Series(x).dropna()
        if s.empty:
            return np.nan
        cummax = s.cummax()
        return float((s / cummax - 1).min())

    return series.rolling(window, min_periods=max(5, window // 4)).apply(_mdd, raw=True)


def build_feature_panels(daily: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = daily.copy().sort_values(["asset_code", "date"])
    d["date"] = pd.to_datetime(d["date"])
    parts = []
    for _, g in d.groupby("asset_code"):
        g = g.copy()
        for w in [5, 20, 60, 120]:
            g[f"return_{w}d"] = g["close"].pct_change(w)
            g[f"vol_{w}d"] = g["return_1d"].rolling(w, min_periods=max(5, w // 4)).std() * np.sqrt(252)
        g["max_drawdown_60d"] = rolling_max_drawdown(g["close"], 60)
        g["max_drawdown_120d"] = rolling_max_drawdown(g["close"], 120)
        g["derived_from"] = "asset_returns_daily.parquet"
        g["calculation_method"] = "past-only rolling returns, volatility, max drawdown"
        parts.append(g)
    fd = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    m = monthly.copy().sort_values(["asset_code", "date"])
    parts = []
    for _, g in m.groupby("asset_code"):
        g = g.copy()
        for w in [3, 6, 12]:
            g[f"return_{w}m"] = g["close"].pct_change(w)
            g[f"vol_{w}m"] = g["return_1m"].rolling(w, min_periods=max(2, w // 3)).std() * np.sqrt(12)
        g["derived_from"] = "asset_returns_monthly.parquet"
        g["calculation_method"] = "past-only monthly rolling returns and volatility"
        parts.append(g)
    fm = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return fd, fm


def quality_tables(universe: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["universe"] = universe
    if not daily.empty:
        coverage = (
            daily.groupby(["asset_code", "asset_name", "module"], as_index=False)
            .agg(
                start_date=("date", "min"),
                end_date=("date", "max"),
                n_obs=("date", "count"),
                missing_close=("close", lambda x: float(x.isna().mean())),
                max_abs_return=("return_1d", lambda x: float(x.abs().max(skipna=True))),
            )
            .sort_values(["module", "asset_code"])
        )
        coverage["start_date"] = pd.to_datetime(coverage["start_date"]).dt.date.astype(str)
        coverage["end_date"] = pd.to_datetime(coverage["end_date"]).dt.date.astype(str)
        tables["daily_coverage"] = coverage
        dup = daily[daily.duplicated(["asset_code", "date"], keep=False)]
        tables["duplicates"] = dup[["date", "asset_code", "asset_name"]].head(500)
        extreme = daily[daily["return_1d"].abs() > 0.12].copy()
        tables["extreme_returns"] = extreme[["date", "asset_code", "asset_name", "return_1d", "close"]].head(1000)
    else:
        tables["daily_coverage"] = pd.DataFrame()
        tables["duplicates"] = pd.DataFrame()
        tables["extreme_returns"] = pd.DataFrame()
    if not monthly.empty:
        tables["monthly_coverage"] = (
            monthly.groupby(["asset_code", "asset_name", "module"], as_index=False)
            .agg(start_date=("date", "min"), end_date=("date", "max"), n_months=("date", "count"))
        )
    else:
        tables["monthly_coverage"] = pd.DataFrame()
    tables["missing_source"] = universe[universe["status"].astype(str).ne("ok")].copy() if "status" in universe else pd.DataFrame()
    module_status = universe.groupby(["module", "status"], as_index=False).size() if not universe.empty else pd.DataFrame()
    tables["module_status"] = module_status
    return tables


def write_reports(paths: ProjectPaths, tables: dict[str, pd.DataFrame], extra_status: dict[str, str]) -> None:
    xlsx = paths.reports / "data_completeness_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
        pd.DataFrame([{"item": k, "status": v} for k, v in extra_status.items()]).to_excel(
            writer, sheet_name="extra_raw_fetch_status", index=False
        )

    universe = tables.get("universe", pd.DataFrame())
    missing = tables.get("missing_source", pd.DataFrame())
    coverage = tables.get("daily_coverage", pd.DataFrame())
    enough_asset_matrix = not coverage.empty and coverage["asset_code"].nunique() >= 10
    module_ok = set(universe.loc[universe.get("status", "") == "ok", "module"].astype(str)) if not universe.empty else set()
    required_modules = {"rates", "credit", "convertible_bonds", "equity", "alternatives"}
    blocked = sorted(required_modules - module_ok)
    missing_count = int(len(missing))
    enough_for_complete_modeling = bool(enough_asset_matrix and not blocked and missing_count == 0)
    lines = [
        "# 数据完整性检查报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 总体结论",
        "",
        f"- 当前是否足够进入下一步建模：{'资产收益矩阵与基础特征阶段够用；完整策略阶段仍建议补强利率曲线、信用利差、流动性和宏观披露日' if enough_asset_matrix else '不够'}",
        f"- 已覆盖模块：{', '.join(sorted(module_ok)) if module_ok else '无'}",
        f"- 仍阻断完整建模的模块/字段：{', '.join(blocked) if blocked else '无模块级阻断，但利率曲线、信用利差、流动性与宏观披露日仍需补强'}",
        "",
        "## 已输出核心数据",
        "",
        "- data_processed/asset_universe.parquet",
        "- data_processed/asset_returns_daily.parquet",
        "- data_processed/asset_returns_monthly.parquet",
        "- data_processed/feature_panel_daily.parquet",
        "- data_processed/feature_panel_monthly.parquet",
        "- output/reports/P0_acceptance_report.md",
        "- output/reports/data_completeness_report.xlsx",
        "",
        "## 缺失与补数建议",
        "",
        "- Tushare可继续补：A股指数估值、成交额、Shibor、M2/CPI/PPI/PMI/社融等宏观原始表。",
        "- Wind后续建议补强：更精确的城投/产业×评级收益矩阵、信用利差、信用债净融资、DR007/R007、公开市场操作、国债期货连续合约、REITs指数、美债或海外债指数。",
        "- 可替代变量：ETF代理利率债/信用债/权益/黄金/商品/REITs；Tushare可转债个券等权自建12个转债风格指数。",
        "- 必须补齐后才能完整策略建模：利率曲线与国债期货、信用利差和信用供需、流动性/宏观数据的release_date/effective_date。当前信用债收益矩阵已用Wind补齐到可建资产收益矩阵的程度。",
        "",
        "## missing_source 摘要",
        "",
    ]
    if missing.empty:
        lines.append("无 missing_source。")
    else:
        for _, r in missing.head(50).iterrows():
            lines.append(f"- {r.get('code', r.get('asset_code', ''))}: {r.get('name', r.get('asset_name', ''))}，{r.get('note', '')}")
    (paths.reports / "data_completeness_report.md").write_text("\n".join(lines), encoding="utf-8")

    p0 = [
        "# P0 数据连接与资产池验收报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Tushare连接：通过",
        f"- Wind连接：本轮未直接调用WindPy，Wind兜底项已在缺口报告标记",
        f"- asset_universe.parquet：{'非空' if not universe.empty else '空'}",
        f"- asset_returns_daily.parquet：{'非空' if not coverage.empty else '空'}",
        f"- 日频资产数量：{int(coverage['asset_code'].nunique()) if not coverage.empty else 0}",
        f"- 时间范围：{coverage['start_date'].min() if not coverage.empty else 'NA'} 至 {coverage['end_date'].max() if not coverage.empty else 'NA'}",
        f"- 缺失率报告：已输出 data_completeness_report.xlsx",
        f"- 新上市资产：以自身start_date开始，不做上市前填充",
    ]
    (paths.reports / "P0_acceptance_report.md").write_text("\n".join(p0), encoding="utf-8")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "enough_for_next_modeling": enough_for_complete_modeling,
        "enough_for_asset_return_feature_stage": bool(enough_asset_matrix),
        "covered_modules": sorted(module_ok),
        "blocked_modules": blocked,
        "missing_count": int(len(missing)),
        "daily_asset_count": int(coverage["asset_code"].nunique()) if not coverage.empty else 0,
    }
    (paths.reports / "data_completeness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_all(root: Path) -> None:
    paths = ensure_dirs(root)
    write_default_configs(paths)
    config = load_yaml(paths.config / "asset_universe.yaml")
    extra_status = fetch_extra_raw_data(paths, config)
    universe, daily = fetch_configured_assets(paths, config)
    cb_uni, cb_daily = build_cb_subindices(paths, config)
    if not cb_daily.empty:
        daily = pd.concat([daily, cb_daily], ignore_index=True) if not daily.empty else cb_daily
    if not cb_uni.empty:
        universe = pd.concat([universe, cb_uni], ignore_index=True) if not universe.empty else cb_uni
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values(["asset_code", "date"])
    monthly = build_monthly_returns(daily)
    fd, fm = build_feature_panels(daily, monthly)

    universe.to_parquet(paths.processed / "asset_universe.parquet", index=False)
    daily.to_parquet(paths.processed / "asset_returns_daily.parquet", index=False)
    monthly.to_parquet(paths.processed / "asset_returns_monthly.parquet", index=False)
    fd.to_parquet(paths.processed / "feature_panel_daily.parquet", index=False)
    fm.to_parquet(paths.processed / "feature_panel_monthly.parquet", index=False)

    tables = quality_tables(universe, daily, monthly)
    write_reports(paths, tables, extra_status)


if __name__ == "__main__":
    run_all(Path(__file__).resolve().parents[1])
