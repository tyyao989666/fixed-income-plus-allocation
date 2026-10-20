from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
OUT = REPORTS / "v1_data_usage_readme.xlsx"

PROFILES = ["low_vol", "balanced", "high_vol"]
MODULE_CN = {
    "rates": "利率债",
    "credit": "信用债",
    "convertible_bonds": "可转债",
    "equity": "权益",
    "alternatives": "另类",
    "cash": "现金",
}

FEATURE_GROUP_CN = {
    "rate": "利率",
    "credit": "信用",
    "funding": "资金面",
    "cb": "可转债",
    "equity": "权益",
    "alt": "另类",
}


def date_coverage(df: pd.DataFrame) -> tuple[str, str]:
    for col in ["date", "month", "as_of_date"]:
        if col in df.columns:
            values = pd.to_datetime(df[col], errors="coerce").dropna()
            if not values.empty:
                return safe_date(values.min()), safe_date(values.max())
    return "", ""


def feature_group(field: str) -> str:
    prefix = field.split("_", 1)[0]
    return FEATURE_GROUP_CN.get(prefix, "状态/元数据")


def feature_description(field: str) -> str:
    rules = [
        ("treasury_y", "国债到期收益率"),
        ("curve_", "期限利差/收益率曲线形态"),
        ("credit_spread", "信用利差、变化或历史分位"),
        ("funding_stress", "资金面压力综合指标"),
        ("r007_dr007_spread", "交易所与银行间资金利差"),
        ("omo_net_injection", "公开市场操作净投放"),
        ("omo_7d_rate", "央行7天逆回购操作利率"),
        ("shibor", "SHIBOR资金利率"),
        ("dr001", "银行间存款类机构隔夜回购利率"),
        ("dr007", "银行间存款类机构7天回购利率"),
        ("r001", "银行间隔夜质押式回购利率"),
        ("r007", "银行间7天质押式回购利率"),
        ("FDR014", "14天存款类机构回购利率"),
        ("FR014", "14天质押式回购利率"),
        ("aaa_cd", "AAA同业存单收益率或代理"),
        ("cb_price_median", "全市场转债价格中位数"),
        ("cb_iv_median", "转债隐含波动率中位数"),
        ("cb_ytm_median", "转债到期收益率中位数"),
        ("cb_premium_median", "转债转股溢价率中位数"),
        ("cb_turnover_amount", "转债市场成交额"),
        ("cb_balance", "转债存量余额"),
        ("low_price_ratio", "低价转债占比"),
        ("high_price", "高价或高溢价转债占比"),
        ("callable_pressure", "转债赎回压力占比"),
        ("csi1000_close", "中证1000指数收盘点位"),
        ("csi1000_amount", "中证1000成交额"),
        ("csi1000_pe", "中证1000市盈率"),
        ("csi1000_pb", "中证1000市净率"),
        ("dividend_close", "红利指数收盘点位"),
        ("dividend_amount", "红利指数成交额"),
        ("dividend_pe", "红利指数市盈率"),
        ("dividend_pb", "红利指数市净率"),
        ("positive_", "申万一级行业上涨比例/市场宽度"),
        ("median_1d_ret", "申万一级行业日收益中位数"),
        ("margin_", "融资融券变化指标"),
        ("rz", "融资融券余额或交易额"),
        ("rq", "融券余额、卖出量或余量"),
        ("us_10y_yield", "美国10年期国债收益率"),
        ("usd_index", "美元指数"),
        ("nanhua_commodity", "南华商品指数"),
        ("gold_au9999", "黄金AU9999价格或收益"),
        ("real_rate_proxy", "实际利率代理指标"),
        ("reits_distribution_yield", "REITs分派率代理"),
    ]
    for token, description in rules:
        if token.lower() in field.lower():
            return description
    if field.endswith("_ret_1d"):
        return "对应资产或指标的日收益率"
    if "change_5d" in field:
        return "5个交易日变化"
    if "change_20d" in field:
        return "20个交易日变化"
    if "pct_2018_to_t" in field:
        return "2018年至当时的历史分位"
    return "市场状态或辅助字段"


def feature_purpose(field: str) -> str:
    group = feature_group(field)
    return {
        "利率": "判断久期方向、曲线形态和利率债风险",
        "信用": "判断信用利差、信用下沉空间和信用风险",
        "资金面": "识别流动性宽松、收紧及资金冲击",
        "可转债": "判断转债估值、价格结构、供给与交易热度",
        "权益": "判断权益估值、宽度、成交和杠杆资金状态",
        "另类": "判断黄金、商品、美债、美元和REITs环境",
    }.get(group, "质量检查、来源追踪或状态标记")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def classify_row(row: pd.Series, universe: pd.DataFrame, returns: pd.DataFrame) -> dict[str, Any]:
    code = str(row["asset_code"])
    module = str(row["module"])
    uni = universe[universe["asset_code"].astype(str).eq(code)]
    ret = returns[returns["asset_code"].astype(str).eq(code)]
    source = ""
    note = ""
    product = ""
    is_proxy = "否"

    if code == "rate_30y_curve_proxy":
        source = "中债30Y国债收益率曲线"
        product = "按30Y收益率曲线合成的恒定久期代理产品"
        is_proxy = "是"
        note = "公式：上月30Y收益率/12 - 18.5×30Y收益率月变化；不是ETF实盘价格。"
    elif code == "cash":
        source = "无"
        product = "现金占位"
        note = "当前配置现金上下限为0，仅保留字段方便检查。"
    elif not uni.empty:
        source = str(uni["data_source"].iloc[0])
        if "tradable_proxy" in uni and pd.notna(uni["tradable_proxy"].iloc[0]):
            product = str(uni["tradable_proxy"].iloc[0])
        elif "ts_code" in uni and pd.notna(uni["ts_code"].iloc[0]):
            product = str(uni["ts_code"].iloc[0])
        else:
            product = str(uni["code"].iloc[0])
        note = str(uni["note"].iloc[0]) if "note" in uni and pd.notna(uni["note"].iloc[0]) else ""
        if "代理" in note or "proxy" in source.lower() or "derived" in source.lower():
            is_proxy = "是"
    elif not ret.empty:
        source = str(ret["data_source"].dropna().iloc[0]) if ret["data_source"].notna().any() else ""
        product = code
        if "proxy" in source.lower() or "derived" in source.lower():
            is_proxy = "是"

    purpose = {
        "rates": "利率久期配置与债券防守/进攻仓位",
        "credit": "信用债底仓与票息/信用利差暴露",
        "convertible_bonds": "转债弹性与股债混合暴露",
        "equity": "权益增强与行业/风格弹性",
        "alternatives": "黄金等另类分散化与宏观对冲",
        "cash": "当前不使用，仅作为零仓位检查项",
    }.get(module, "")

    return {
        "大类": MODULE_CN.get(module, module),
        "资产代码": code,
        "资产名称": str(row["asset_name"]),
        "产品/口径": product,
        "数据来源": source,
        "是否代理": is_proxy,
        "数据起始": safe_date(row["data_start"]),
        "数据结束": safe_date(row["data_end"]),
        "低波平均权重": float(row.get("low_vol_avg_weight", 0.0)),
        "中波平均权重": float(row.get("balanced_avg_weight", 0.0)),
        "高波平均权重": float(row.get("high_vol_avg_weight", 0.0)),
        "低波最新权重": float(row.get("low_vol_latest_weight", 0.0)),
        "中波最新权重": float(row.get("balanced_latest_weight", 0.0)),
        "高波最新权重": float(row.get("high_vol_latest_weight", 0.0)),
        "用途": purpose,
        "备注": note,
    }


def build_asset_table() -> pd.DataFrame:
    frames = []
    for profile in PROFILES:
        path = PROCESSED / f"v1_{profile}_target_weights_monthly.parquet"
        w = pd.read_parquet(path)
        w["date"] = pd.to_datetime(w["date"], errors="coerce")
        agg = (
            w.groupby(["asset_code", "asset_name", "module"], as_index=False)
            .agg(start=("date", "min"), end=("date", "max"), avg_weight=("weight", "mean"), latest_weight=("weight", "last"))
        )
        agg["profile"] = profile
        frames.append(agg)
    all_w = pd.concat(frames, ignore_index=True)
    pivot = all_w.pivot_table(
        index=["asset_code", "asset_name", "module"],
        columns="profile",
        values=["avg_weight", "latest_weight"],
        aggfunc="first",
    )
    pivot.columns = [f"{profile}_{metric}" for metric, profile in pivot.columns]
    pivot = pivot.reset_index()
    universe = pd.read_parquet(PROCESSED / "asset_universe.parquet")
    returns = pd.read_parquet(PROCESSED / "asset_returns_monthly.parquet")
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    coverage = (
        returns.dropna(subset=["return_1m"])
        .groupby("asset_code", as_index=False)
        .agg(data_start=("date", "min"), data_end=("date", "max"))
    )
    out = pivot.merge(coverage, on="asset_code", how="left")
    for profile in PROFILES:
        for metric in ["avg_weight", "latest_weight"]:
            col = f"{profile}_{metric}"
            if col not in out:
                out[col] = 0.0
            out[col] = out[col].fillna(0.0)
    rows = [classify_row(row, universe, returns) for _, row in out.sort_values(["module", "asset_code"]).iterrows()]
    return pd.DataFrame(rows)


def build_overview_table(cfg: dict[str, Any]) -> pd.DataFrame:
    metrics_rows = []
    for profile in PROFILES:
        m = pd.read_csv(REPORTS / f"v1_{profile}_backtest_metrics.csv").set_index("metric")["value"]
        metrics_rows.append(
            {
                "产品线": profile,
                "年化收益": float(m["annual_return"]),
                "年化波动": float(m["annual_vol"]),
                "Sharpe": float(m["sharpe"]),
                "最大回撤": float(m["max_drawdown"]),
                "累计收益": float(m["total_return"]),
                "月均换手": float(m["avg_turnover"]),
            }
        )
    return pd.DataFrame(metrics_rows)


def build_pipeline_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "步骤": 1,
                "环节": "基础资产数据",
                "输入": "config/asset_universe.yaml；Tushare/Wind/本地转债数据",
                "处理": "拉取ETF、商品指数、REITs、信用代理和转债子指数；统一资产代码与日期",
                "输出": "asset_universe.parquet；asset_returns_daily.parquet；asset_returns_monthly.parquet",
                "作用": "提供所有可配置资产的价格和收益率",
                "频率": "日频/月频",
            },
            {
                "步骤": 2,
                "环节": "30Y利率债代理",
                "输入": "rate_curves_daily_v43.parquet 的 treasury_y30",
                "处理": "按上月收益率/12 - 18.5×收益率月变化构造恒定久期收益",
                "输出": "rate_30y_curve_proxy_monthly.parquet",
                "作用": "补齐2018年以来超长久期利率债历史",
                "频率": "月频",
            },
            {
                "步骤": 3,
                "环节": "市场状态卡",
                "输入": "feature_panel_daily_v43.parquet；宏观、利率、信用、转债、权益、另类特征",
                "处理": "按规则生成利率、信用、转债、权益、另类和风险预算状态",
                "输出": "market_state_cards_daily_v1.parquet",
                "作用": "把市场环境转成可解释状态，不直接给仓位",
                "频率": "日频状态/月末使用",
            },
            {
                "步骤": 4,
                "环节": "Market View",
                "输入": "market_state_cards_daily_v1.parquet",
                "处理": "机械规则或GLM生成相对观点；禁止现金观点和直接仓位输出",
                "输出": "market_skill_views_daily_v1.parquet；ep_views_daily_v1.parquet",
                "作用": "给风险预算底仓提供有限方向倾斜",
                "频率": "日频生成/月末生效",
            },
            {
                "步骤": 5,
                "环节": "iTransformer预测",
                "输入": "itransformer_dataset_daily_path.parquet；日频多资产路径与状态特征",
                "处理": "用过去日频路径预测下一月各袖套收益/排序",
                "输出": "itransformer_predictions_daily_path_monthly.parquet",
                "作用": "提供袖套级预测倾斜，不直接决定最终仓位",
                "频率": "日频输入/月频输出",
            },
            {
                "步骤": 6,
                "环节": "风险预算底仓",
                "输入": "asset_returns_monthly.parquet；产品目标风险贡献；大类上下限",
                "处理": "滚动36个月协方差，最少12个月历史，求解目标风险贡献权重",
                "输出": "v1_*_risk_budget_base_monthly.parquet",
                "作用": "确定每个产品的可解释大类底仓",
                "频率": "月频",
            },
            {
                "步骤": 7,
                "环节": "EP/skfolio优化",
                "输入": "风险预算底仓；Market View；趋势；iTransformer；大类上下限和换手约束",
                "处理": "合成后验收益，并由skfolio MeanRisk/CLARABEL约束化落仓",
                "输出": "v1_*_sleeve_weights_monthly.parquet；v1_*_ep_posterior_monthly.parquet；optimizer diagnostics",
                "作用": "生成最终大类权重并保留优化诊断",
                "频率": "月频",
            },
            {
                "步骤": 8,
                "环节": "袖套内资产分配",
                "输入": "最终大类权重；当月可用资产；资产级iTransformer；产品专属资产偏好",
                "处理": "在每个袖套内部配置具体资产；高波偏创业板/中证1000/科创50及黄金商品",
                "输出": "v1_*_target_weights_monthly.parquet",
                "作用": "形成可执行的具体资产持仓",
                "频率": "月频",
            },
            {
                "步骤": 9,
                "环节": "T+1回测",
                "输入": "v1_*_target_weights_monthly.parquet；asset_returns_monthly.parquet；5bp交易成本",
                "处理": "权重滞后一期执行，计算换手、成本、月收益、净值和回撤",
                "输出": "v1_*_backtest_nav.csv；v1_*_backtest_metrics.csv",
                "作用": "评价产品收益、风险和交易成本后表现",
                "频率": "月频",
            },
            {
                "步骤": 10,
                "环节": "归因与可视化",
                "输入": "净值、最终权重、资产收益和优化诊断",
                "处理": "计算大类/资产/月度/年度贡献并生成图表",
                "输出": "attribution CSV；v1_visual_report.html；output/figures/v1/*.png",
                "作用": "解释收益来源、持仓变化和策略风险",
                "频率": "全样本汇总",
            },
        ]
    )


def build_data_file_table() -> pd.DataFrame:
    specs = [
        ("asset_universe.parquet", "资产主表", "资产代码、名称、大类、数据源、交易代理及备注", "资产映射和数据拉取"),
        ("asset_returns_daily.parquet", "资产日行情", "OHLC、成交量、成交额、日收益率", "日频特征、月度收益和回测"),
        ("asset_returns_monthly.parquet", "资产月收益", "月末价格、月收益率及资产分类", "风险预算、优化器和月度回测"),
        ("feature_panel_daily_v43.parquet", "日频综合特征", "利率、信用、资金面、转债、权益和另类共107列", "MarketSkill和日频路径iTransformer"),
        ("feature_panel_monthly_v43.parquet", "月频综合特征", "日频特征按真实月末可见值汇总", "月频模型、状态判断和审计"),
        ("market_state_cards_daily_v1.parquet", "市场状态卡", "CR/R/CB/EQ/ALT/RB状态、风险标签和可投池", "Market View输入"),
        ("itransformer_dataset_daily_path.parquet", "日频路径训练集", "过去120个交易日的多变量序列及下一月标签", "日频路径iTransformer训练与预测"),
        ("itransformer_predictions_daily_path_monthly.parquet", "袖套预测", "五大袖套下一月预测收益、排序概率", "约束内预测倾斜"),
        ("asset_itransformer_predictions_monthly.parquet", "资产级预测", "袖套内部各资产预测收益和概率", "袖套内具体资产分配"),
        ("market_skill_views_daily_v1.parquet", "市场观点", "机械规则或GLM产生的相对观点和置信度", "EP后验收益输入"),
        ("ep_views_daily_v1.parquet", "EP标准观点", "映射为EP可用的源资产、目标资产和置信度", "EP/skfolio优化"),
    ]
    rows = []
    for filename, category, content, purpose in specs:
        path = PROCESSED / filename
        if path.exists():
            df = pd.read_parquet(path)
            start, end = date_coverage(df)
            rows.append(
                {
                    "文件": filename,
                    "类别": category,
                    "行数": len(df),
                    "列数": len(df.columns),
                    "数据起始": start,
                    "数据结束": end,
                    "日期口径": "月末标签；未完成月份不得进入正式回测" if "monthly" in filename else "实际交易日/观测日",
                    "主要内容": content,
                    "当前用途": purpose,
                }
            )
        else:
            rows.append({"文件": filename, "类别": category, "行数": 0, "列数": 0, "数据起始": "", "数据结束": "", "日期口径": "", "主要内容": content, "当前用途": purpose + "（文件当前不存在）"})
    return pd.DataFrame(rows)


def build_asset_field_table() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "asset_returns_daily.parquet")
    descriptions = {
        "date": ("交易日期", "主键/时间对齐", "是"),
        "asset_code": ("策略内部资产代码", "资产映射和持仓关联", "是"),
        "code": ("数据源代码", "数据追溯", "间接"),
        "asset_name": ("资产中文名称", "报表展示", "间接"),
        "asset_class": ("资产类别", "资产分类", "是"),
        "module": ("风险袖套", "风险预算和归因", "是"),
        "sub_module": ("细分资产类别", "细分归因", "间接"),
        "ts_code": ("Tushare/Wind证券代码", "数据追溯", "间接"),
        "open": ("开盘价", "已保存；当前月度回测未直接使用", "否"),
        "high": ("最高价", "已保存；当前模型未直接使用", "否"),
        "low": ("最低价", "已保存；当前模型未直接使用", "否"),
        "close": ("收盘价/指数点位", "计算日收益和月收益", "是"),
        "vol": ("成交量", "流动性研究储备；部分源可能缺失", "有限"),
        "amount": ("成交额", "流动性及交易热度特征", "有限"),
        "return_1d": ("由收盘价计算的日收益率", "日频路径、动量、波动和回撤", "是"),
        "data_source": ("数据来源", "代理披露和审计", "是"),
        "fetch_time": ("抓取时间", "数据版本追踪", "间接"),
        "member_count": ("指数/组合成分数量", "合成指数质量检查", "有限"),
    }
    rows = []
    for field in df.columns:
        series = df[field]
        valid = series.notna()
        start = end = ""
        if field != "date" and "date" in df and valid.any():
            dates = pd.to_datetime(df.loc[valid, "date"], errors="coerce").dropna()
            if not dates.empty:
                start, end = safe_date(dates.min()), safe_date(dates.max())
        description, purpose, used = descriptions.get(field, ("行情或元数据字段", "辅助分析", "有限"))
        rows.append(
            {
                "字段": field,
                "中文含义": description,
                "数据类型": str(series.dtype),
                "非空数量": int(valid.sum()),
                "缺失率": float(1.0 - valid.mean()),
                "有效起始": start,
                "有效结束": end,
                "当前用途": purpose,
                "当前直接使用": used,
            }
        )
    return pd.DataFrame(rows)


def build_feature_field_table() -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED / "feature_panel_daily_v43.parquet")
    monthly_path = PROCESSED / "feature_panel_monthly_v43.parquet"
    monthly = pd.read_parquet(monthly_path) if monthly_path.exists() else pd.DataFrame()
    rows = []
    metadata_fields = {"date", "month", "source", "quality_flag", "frequency", "is_proxy", "note"}
    for field in daily.columns:
        if field == "date":
            continue
        series = daily[field]
        valid = series.notna()
        dates = pd.to_datetime(daily.loc[valid, "date"], errors="coerce").dropna() if valid.any() else pd.Series(dtype="datetime64[ns]")
        is_proxy = "是" if "proxy" in field.lower() else "否"
        if field in metadata_fields:
            is_proxy = "元数据"
        rows.append(
            {
                "模块": feature_group(field),
                "字段": field,
                "中文含义": feature_description(field),
                "日频非空数": int(valid.sum()),
                "日频缺失率": float(1.0 - valid.mean()),
                "有效起始": safe_date(dates.min()) if not dates.empty else "",
                "有效结束": safe_date(dates.max()) if not dates.empty else "",
                "月频字段存在": "是" if field in monthly.columns else "否",
                "是否代理": is_proxy,
                "当前用途": feature_purpose(field),
                "进入模块": "MarketSkill + 日频路径iTransformer" if field not in metadata_fields else "数据质量与审计",
            }
        )
    return pd.DataFrame(rows)


def build_itransformer_feature_table() -> pd.DataFrame:
    metadata_files = {
        "原日频路径": REPORTS / "itransformer_daily_dataset_metadata.json",
        "OHLCV增强候选": REPORTS / "itransformer_daily_ohlcv_metadata.json",
    }
    feature_sets: dict[str, set[str]] = {}
    for version, path in metadata_files.items():
        if path.exists():
            feature_sets[version] = set(json.loads(path.read_text(encoding="utf-8")).get("feature_columns", []))
    all_features = sorted(set().union(*feature_sets.values())) if feature_sets else []
    rows = []
    for field in all_features:
        rows.append(
            {
                "特征": field,
                "原日频路径": "是" if field in feature_sets.get("原日频路径", set()) else "否",
                "OHLCV增强候选": "是" if field in feature_sets.get("OHLCV增强候选", set()) else "否",
                "特征类别": "OHLCV派生" if "_ohlcv_" in field else feature_group(field),
                "说明": feature_description(field) if "_ohlcv_" not in field else "由开高低收、成交量或成交额构造的尺度无关特征",
                "正式状态": "正式模型使用" if field in feature_sets.get("原日频路径", set()) else "增强实验使用，尚未替换正式模型",
            }
        )
    return pd.DataFrame(rows)


def build_product_table(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for profile in PROFILES:
        spec = cfg["product"]["profiles"][profile]
        for sleeve in ["rates", "credit", "convertible_bonds", "equity", "alternatives", "cash"]:
            bounds = spec["sleeve_bounds"].get(sleeve, [0.0, 0.0])
            rows.append(
                {
                    "产品": profile,
                    "大类": MODULE_CN.get(sleeve, sleeve),
                    "目标风险贡献": spec.get("target_risk_contribution", {}).get(sleeve, 0.0),
                    "权重下限": float(bounds[0]),
                    "权重上限": float(bounds[1]),
                    "定位": {"low_vol": "稳健夏普优先", "balanced": "收益风险平衡", "high_vol": "回撤约束下的绝对收益增强"}[profile],
                }
            )
    return pd.DataFrame(rows)


def build_checks_table(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for profile in PROFILES:
        weights = pd.read_parquet(PROCESSED / f"v1_{profile}_target_weights_monthly.parquet")
        sums = weights.groupby("date")["weight"].sum()
        cash = weights.loc[weights["asset_code"].eq("cash"), "weight"]
        rows.extend(
            [
                {"产品": profile, "检查项": "权重合计最小值", "结果": float(sums.min()), "状态": "通过" if abs(float(sums.min()) - 1.0) <= 1e-6 else "异常"},
                {"产品": profile, "检查项": "权重合计最大值", "结果": float(sums.max()), "状态": "通过" if abs(float(sums.max()) - 1.0) <= 1e-6 else "异常"},
                {"产品": profile, "检查项": "现金最大权重", "结果": float(cash.max()) if not cash.empty else 0.0, "状态": "通过" if cash.empty or float(cash.max()) <= 1e-12 else "异常"},
                {"产品": profile, "检查项": "最新权重日期", "结果": safe_date(weights["date"].max()), "状态": "通过"},
            ]
        )
    return pd.DataFrame(rows)


def build_outputs_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"文件": "v1_*_backtest_nav.csv", "内容": "月度收益、换手、交易成本、净值", "用途": "PnL/回撤图和绩效评估"},
            {"文件": "v1_*_backtest_metrics.csv", "内容": "年化收益、波动、Sharpe、回撤等", "用途": "产品线对比"},
            {"文件": "v1_*_attribution_by_sleeve.csv", "内容": "大类总贡献和平均实际权重", "用途": "解释收益来源"},
            {"文件": "v1_*_monthly_attribution.csv", "内容": "月度大类贡献", "用途": "累计贡献图和阶段归因"},
            {"文件": "v1_*_ep_signals.csv", "内容": "进入EP后验收益的MarketSkill/趋势/iTransformer信号", "用途": "解释优化器为什么倾斜"},
            {"文件": "v1_*_optimizer_diagnostics.csv", "内容": "优化器状态、主动偏离、袖套换手、预期收益、组合月波动", "用途": "检查EP/skfolio是否正常收敛和是否越界"},
            {"文件": "v1_*_ep_posterior_monthly.parquet", "内容": "每月袖套后验收益估计", "用途": "复核EP优化输入"},
            {"文件": "v1_visual_report.html", "内容": "净值、持仓、换手、归因可视化汇总", "用途": "投研查看结果"},
            {"文件": "output/figures/v1/*.png", "内容": "可视化图片", "用途": "报告/PPT引用"},
            {"文件": "v1_full_pipeline_manifest.json", "内容": "一键流程运行清单", "用途": "检查每个产品线是否成功生成"},
        ]
    )


def add_df(ws, df: pd.DataFrame, start_row: int = 1, start_col: int = 1) -> None:
    headers = list(df.columns)
    for j, header in enumerate(headers, start_col):
        ws.cell(start_row, j, header)
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, value in enumerate(row, start_col):
            ws.cell(i, j, value)


def style_sheet(ws, title_rows: int = 1) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    sub_fill = PatternFill("solid", fgColor="D9EAF7")
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = border
            if isinstance(cell.value, float):
                cell.number_format = "0.00%"
    for row in range(1, title_rows + 1):
        for cell in ws[row]:
            cell.fill = header_fill if row == 1 else sub_fill
            cell.font = header_font if row == 1 else Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = f"A{title_rows + 1}"
    ws.auto_filter.ref = ws.dimensions
    for col_cells in ws.columns:
        max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max(max_len + 2, 10), 36)


def main() -> None:
    cfg = read_yaml(ROOT / "config" / "v1_strategy_config.yaml")
    REPORTS.mkdir(parents=True, exist_ok=True)

    overview = build_overview_table(cfg)
    assets = build_asset_table()
    data_files = build_data_file_table()
    asset_fields = build_asset_field_table()
    feature_fields = build_feature_field_table()
    itransformer_features = build_itransformer_feature_table()
    pipeline = build_pipeline_table()
    products = build_product_table(cfg)
    outputs = build_outputs_table()
    checks = build_checks_table(cfg)

    wb = Workbook()
    ws = wb.active
    ws.title = "总览"
    summary = pd.DataFrame(
        [
            {"项目": "回测窗口", "说明": f"{cfg['rebalance']['start_date']} 至 {cfg['rebalance']['end_date']}"},
            {"项目": "执行口径", "说明": "月频再平衡，信号滞后1期，5bp换手成本"},
            {"项目": "现金", "说明": "当前现金上下限为0，不使用现金仓位"},
            {"项目": "30Y口径", "说明": "使用中债30Y国债收益率曲线合成30Y国债曲线代理产品"},
            {"项目": "产品线", "说明": "low_vol / balanced / high_vol；正式高波为回撤约束下的绝对收益增强版"},
            {"项目": "高波新增资产", "说明": "创业板ETF 159915.SZ；中证1000ETF 512100.SH；同时保留科创50ETF"},
        ]
    )
    add_df(ws, summary, 1, 1)
    add_df(ws, overview, 9, 1)
    ws["A8"] = "产品线最新回测指标"
    ws["A8"].font = Font(bold=True, color="1F4E78")
    style_sheet(ws)

    ws_assets = wb.create_sheet("资产与产品口径")
    add_df(ws_assets, assets)
    style_sheet(ws_assets)

    ws_files = wb.create_sheet("核心数据文件")
    add_df(ws_files, data_files)
    style_sheet(ws_files)

    ws_asset_fields = wb.create_sheet("资产行情字段")
    add_df(ws_asset_fields, asset_fields)
    style_sheet(ws_asset_fields)

    ws_features = wb.create_sheet("模型特征字段")
    add_df(ws_features, feature_fields)
    style_sheet(ws_features)

    ws_itransformer = wb.create_sheet("iTransformer实际输入")
    add_df(ws_itransformer, itransformer_features)
    style_sheet(ws_itransformer)

    ws_pipeline = wb.create_sheet("全流程输入输出")
    add_df(ws_pipeline, pipeline)
    style_sheet(ws_pipeline)

    ws_product = wb.create_sheet("产品参数")
    add_df(ws_product, products)
    style_sheet(ws_product)

    ws_out = wb.create_sheet("输出文件说明")
    add_df(ws_out, outputs)
    style_sheet(ws_out)

    ws_checks = wb.create_sheet("运行检查")
    add_df(ws_checks, checks)
    style_sheet(ws_checks)

    save_path = OUT
    try:
        wb.save(save_path)
    except PermissionError:
        save_path = OUT.with_name("v1_data_usage_readme_enhanced.xlsx")
        wb.save(save_path)

    check = load_workbook(save_path, data_only=False)
    expected = {"总览", "资产与产品口径", "核心数据文件", "资产行情字段", "模型特征字段", "iTransformer实际输入", "全流程输入输出", "产品参数", "输出文件说明", "运行检查"}
    if set(check.sheetnames) != expected:
        raise RuntimeError(f"Unexpected sheets: {check.sheetnames}")
    manifest = {
        "workbook": str(save_path.relative_to(ROOT)),
        "sheets": check.sheetnames,
        "asset_rows": len(assets),
        "data_file_rows": len(data_files),
        "asset_field_rows": len(asset_fields),
        "feature_field_rows": len(feature_fields),
        "itransformer_feature_rows": len(itransformer_features),
        "pipeline_rows": len(pipeline),
        "product_rows": len(products),
        "check_rows": len(checks),
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
