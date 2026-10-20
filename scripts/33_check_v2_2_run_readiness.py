from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_TS = ROOT / "data_raw" / "tushare"
RAW_WIND = ROOT / "data_raw" / "wind"
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
CONFIG = ROOT / "config"


def parquet_range(path: Path, date_candidates=("date", "trade_date", "cal_date", "month", "end_date", "period")) -> dict:
    if not path.exists():
        return {"exists": False, "rows": 0, "start": None, "end": None, "columns": ""}
    df = pd.read_parquet(path)
    date_col = next((c for c in date_candidates if c in df.columns), None)
    start = end = None
    if date_col:
        s = pd.to_datetime(df[date_col], errors="coerce")
        if s.notna().any():
            start = s.min()
            end = s.max()
    return {
        "exists": True,
        "rows": len(df),
        "start": start,
        "end": end,
        "columns": ",".join(map(str, df.columns[:30])),
    }


def json_rows(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "rows": 0, "columns": ""}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"exists": True, "rows": 0, "columns": f"parse_error:{exc}"}

    def count(x):
        if isinstance(x, dict) and "rows" in x:
            return len(x.get("rows") or [])
        if isinstance(x, dict) and "data" in x:
            return sum(count(i) for i in x.get("data") or [])
        if isinstance(x, list):
            return sum(count(i) for i in x)
        return 1 if x is not None else 0

    def cols(x):
        if isinstance(x, dict) and "columns" in x:
            return ",".join(str(c.get("name", "")) for c in x.get("columns") or [])
        if isinstance(x, dict) and "data" in x:
            out = []
            for i in x.get("data") or []:
                c = cols(i)
                if c:
                    out.append(c)
            return " | ".join(out[:3])
        if isinstance(x, list):
            out = []
            for i in x:
                c = cols(i)
                if c:
                    out.append(c)
            return " | ".join(out[:3])
        return ""

    return {"exists": True, "rows": count(payload), "columns": cols(payload)}


def status_label(source_ready: bool, standardized: bool, blocker: bool = False) -> str:
    if standardized and not blocker:
        return "ready"
    if source_ready and not standardized:
        return "raw_needs_parse"
    if source_ready and blocker:
        return "partial_verify"
    return "missing_or_blocked"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    asset_universe = parquet_range(PROCESSED / "asset_universe.parquet")
    asset_returns = parquet_range(PROCESSED / "asset_returns_daily.parquet")
    feature_daily = parquet_range(PROCESSED / "feature_panel_daily.parquet")
    feature_monthly = parquet_range(PROCESSED / "feature_panel_monthly.parquet")
    cb_monthly = parquet_range(PROCESSED / "existing_cb_market_features_monthly.parquet")

    checks = []
    def add(module, requirement, source_files, standardized_files, status, conclusion, action):
        checks.append(
            {
                "module": module,
                "requirement": requirement,
                "source_files": source_files,
                "standardized_files": standardized_files,
                "status": status,
                "conclusion": conclusion,
                "next_action": action,
            }
        )

    add(
        "P0资产收益矩阵",
        "五大资产模块34个资产、日/月频收益",
        "data_raw/tushare + data_raw/wind",
        "asset_universe.parquet; asset_returns_daily.parquet; asset_returns_monthly.parquet",
        "ready" if asset_universe["exists"] and asset_returns["exists"] else "missing_or_blocked",
        f"已覆盖34个资产，日频{asset_returns['start']}至{asset_returns['end']}；但部分ETF/转债子指数从2020以后开始。",
        "可跑静态基准和资产级风险特征；新上市资产需按start_date入池。",
    )
    add(
        "资产自身特征",
        "return/momentum/vol/max_drawdown等滚动特征",
        "asset_returns_daily.parquet",
        "feature_panel_daily.parquet; feature_panel_monthly.parquet",
        "ready" if feature_daily["exists"] and feature_monthly["exists"] else "missing_or_blocked",
        "已有return_5d/20d/60d/120d、vol、max_drawdown；price_position可继续派生。",
        "补几个命名字段即可，不阻断数据底座。",
    )
    add(
        "可转债状态CB",
        "转债余额、价格、转股溢价率、纯债溢价率、YTM、换手、隐波、低/高价占比",
        r"C:\Users\lenovo\Desktop\数据库下载\修改\Data\可转债数据",
        "existing_cb_market_features_daily.parquet; existing_cb_market_features_monthly.parquet",
        "ready" if cb_monthly["exists"] else "missing_or_blocked",
        f"已接入现有可转债库，{cb_monthly['start']}至{cb_monthly['end']}，月频{cb_monthly['rows']}行。",
        "可作为CB模块主口径；Wind转债自然语言表只作为备查。",
    )
    add(
        "利率债R",
        "利率债资产收益、国债收益率曲线1/3/5/10/30Y、期限利差、30Y风险",
        "Tushare债券ETF/基金；Wind analytics_treasury_yield_curve",
        "资产收益已标准化；收益率曲线暂无rate_curve_daily/monthly.parquet",
        status_label((RAW_WIND / "analytics_treasury_yield_curve_20180101_20260603.json").exists(), False),
        "资产收益可跑；R状态里的期限利差/曲线陡峭度还停在Wind原始JSON。",
        "需要解析Wind收益率曲线JSON，生成rate_curve_monthly.parquet；国债期货可先不做或标记缺失。",
    )
    add(
        "信用债CR",
        "城投/产业×评级收益、信用利差、信用下沉、到期压力、净融资、取消发行、评级下调/违约",
        "Wind信用指数K线；Wind credit_spreads；Wind违约/到期/净融资探测",
        "信用债收益矩阵已标准化；信用利差/尾部事件未标准化",
        "raw_needs_parse",
        "信用债资产收益已经完整；CR状态需要的利差和信用尾部变量仍需解析。净融资/取消发行/评级下调目前有权限或口径问题。",
        "先用信用利差+违约规模+到期压力跑CR简版；净融资/取消发行/评级下调标记missing_source或用Wind继续补。",
    )
    add(
        "权益EQ",
        "沪深300/中证500/中证1000/创业板/万得全A/红利、估值、成交额、两融、扩散度",
        "Tushare index_daily/index_dailybasic/margin；Wind万得全A/红利/VIX",
        "指数行情原始表齐；估值部分齐；扩散度未标准化",
        "raw_needs_parse",
        "指数行情和主要估值基本有；中证1000、红利和中证转债dailybasic为空；ERP和扩散度需计算。",
        "可先用HS300/CSI500/创业板/SSE估值分位+两融+权益资产动量跑EQ简版。",
    )
    add(
        "流动性LIQ",
        "DR007/R007/Shibor/AAA同业存单/OMO/政策利率/流动性冲击",
        "Tushare Shibor；Wind DR007/R007、OMO、AAA CD 1Y",
        "暂无liquidity_monthly.parquet",
        "raw_needs_parse",
        "Shibor已有结构化Tushare表；DR007/R007和OMO为Wind原始JSON；AAA CD只找到100行analytics结果。",
        "解析DR007/R007/OMO；AAA CD不足时用Shibor3M或DR007均值代理。",
    )
    add(
        "宏观MACRO",
        "PMI/M2/CPI/PPI/社融/GDP/工增/固投/出口及披露滞后",
        "Tushare cn_m/cpi/ppi/pmi/sf_month/gdp；Wind工增/固投/出口",
        "暂无macro_factors_monthly.parquet",
        "raw_needs_parse",
        "主要宏观原始表已在；工增/固投/出口通过Wind找到了；release/effective_date还未结构化。",
        "生成macro_factors_monthly.parquet，并先用保守1个月滞后避免未来函数。",
    )
    add(
        "另类ALT/海外",
        "黄金、商品、REITs、美债、VIX、美元指数、美债10Y/实际利率",
        "Tushare ETF代理；Wind VIX/南华商品/黄金/美元/美债10Y/实际利率",
        "资产收益已标准化；海外状态变量未标准化",
        "raw_needs_parse",
        "另类资产回报可跑；ALT状态里的美元、美债实际利率、VIX等需解析或用ETF代理简化。",
        "若只是跑通，用黄金/商品/REITs/美债ETF收益+VIX即可；海外利率后续补标准表。",
    )
    add(
        "Market State Card",
        "E/B/CB/CR/R/EQ/ALT/RB/liquidity_shock统一状态卡",
        "上述全部raw/processed",
        "market_state_cards.parquet尚不存在",
        "missing_or_blocked",
        "这是当前真正的跑通阻断点：数据有不少，但状态卡还没有构建。",
        "下一步应先做state_engine_features_monthly.parquet和market_state_cards.parquet，不进入策略优化。",
    )

    required_configs = [
        "state_engine_config.yaml",
        "state_thresholds.yaml",
        "state_action_rules.yaml",
        "product_constraints.yaml",
        "liquidity_shock_rules.yaml",
        "risk_budget_rules.yaml",
        "market_state_card_schema.yaml",
        "scenario_thresholds.yaml",
    ]
    config_rows = []
    for name in required_configs:
        p = CONFIG / name
        config_rows.append({"config": name, "exists": p.exists(), "length": p.stat().st_size if p.exists() else 0})
    config_df = pd.DataFrame(config_rows)

    df = pd.DataFrame(checks)
    ready_count = (df["status"] == "ready").sum()
    raw_count = (df["status"] == "raw_needs_parse").sum()
    blocked_count = (df["status"] == "missing_or_blocked").sum()

    xlsx = REPORTS / "v2_2_run_readiness_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="readiness", index=False)
        config_df.to_excel(writer, sheet_name="config_check", index=False)
        pd.DataFrame(
            [
                {"file": "asset_universe", **asset_universe},
                {"file": "asset_returns_daily", **asset_returns},
                {"file": "feature_panel_daily", **feature_daily},
                {"file": "feature_panel_monthly", **feature_monthly},
                {"file": "existing_cb_market_features_monthly", **cb_monthly},
            ]
        ).to_excel(writer, sheet_name="core_file_ranges", index=False)

    md = REPORTS / "v2_2_run_readiness_report.md"
    lines = [
        "# v2.2 项目数据跑通检查报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 结论",
        "",
        f"- 可直接用：{ready_count} 项",
        f"- 已有原始数据但需解析/标准化：{raw_count} 项",
        f"- 当前阻断：{blocked_count} 项",
        "",
        "当前结论：**数据源层面已经覆盖大部分，但 MD 的五资产状态引擎还不能完整直接跑通**。P0资产收益矩阵可以跑，CB模块现在也可以跑；真正需要补的是利率曲线、信用利差/尾部、流动性、宏观、海外变量的标准化，以及最终 Market State Card。",
        "",
        "## 模块明细",
        "",
    ]
    for _, r in df.iterrows():
        lines.extend(
            [
                f"### {r['module']} - {r['status']}",
                f"- 要求：{r['requirement']}",
                f"- 现有来源：{r['source_files']}",
                f"- 标准化文件：{r['standardized_files']}",
                f"- 判断：{r['conclusion']}",
                f"- 下一步：{r['next_action']}",
                "",
            ]
        )
    lines.extend(["## 配置缺口", ""])
    for _, r in config_df.iterrows():
        mark = "OK" if r["exists"] and r["length"] > 20 else "缺/空"
        lines.append(f"- {r['config']}: {mark}")
    lines.extend(
        [
            "",
            "## 建议执行顺序",
            "",
            "1. 先把 Wind/Tushare 原始表解析成 `state_engine_features_monthly.parquet`。",
            "2. 用简化规则生成 `market_state_cards.parquet`，先保证 v2.2 状态引擎可跑。",
            "3. 对仍缺的信用净融资、取消发行、评级下调，先标记 `missing_source`，不阻断主流程。",
            "4. 暂不进入 iTransformer、EP、skfolio 和回测，先完成数据底座闭环。",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")
    print(md)
    print(xlsx)
    print(df[["module", "status", "conclusion"]].to_string(index=False))


if __name__ == "__main__":
    main()
