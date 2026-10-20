from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CB_ROOT = Path(r"C:\Users\lenovo\Desktop\数据库下载\修改\Data\可转债数据")
STATE_PATH = Path(r"C:\Users\lenovo\Desktop\可转债数据库生成\state.json")
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


PANELS = {
    "cb_close": CB_ROOT / "1_Base_Data" / "转债收盘价_tushare.parquet",
    "cb_amount": CB_ROOT / "1_Base_Data" / "成交额_tushare.parquet",
    "cb_balance": CB_ROOT / "1_Base_Data" / "债券余额.parquet",
    "cb_parity": CB_ROOT / "1_Base_Data" / "平价_tushare.parquet",
    "cb_conversion_premium": CB_ROOT / "1_Base_Data" / "平价溢价率_tushare.parquet",
    "cb_bond_premium": CB_ROOT / "1_Base_Data" / "纯债溢价率_tushare.parquet",
    "cb_bond_value": CB_ROOT / "1_Base_Data" / "纯债价值_tushare.parquet",
    "cb_ytm": CB_ROOT / "1_Base_Data" / "YTM.parquet",
    "cb_turnover": CB_ROOT / "2_Derived_Data" / "换手率.parquet",
    "cb_iv": CB_ROOT / "2_Derived_Data" / "隐含波动率_bt.parquet",
}


def read_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    date_col = next((c for c in ["trade_date", "date", "datetime", "__index_level_0__"] if c in df.columns), None)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df = df.drop(columns=[date_col])
    else:
        dates = pd.to_datetime(df.index, errors="coerce")
    df.index = dates
    df = df.loc[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def wavg(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    panels: dict[str, pd.DataFrame] = {}
    coverage = []
    for name, path in PANELS.items():
        if not path.exists():
            coverage.append({"panel": name, "path": str(path), "exists": False, "rows": 0, "cols": 0, "start": None, "end": None})
            continue
        df = read_panel(path)
        panels[name] = df
        coverage.append(
            {
                "panel": name,
                "path": str(path),
                "exists": True,
                "rows": len(df),
                "cols": df.shape[1],
                "start": df.index.min(),
                "end": df.index.max(),
                "missing_ratio": float(df.isna().mean().mean()) if df.size else np.nan,
            }
        )

    close = panels["cb_close"]
    amount = panels.get("cb_amount", close * np.nan)
    balance = panels.get("cb_balance", close * np.nan)
    conv_premium = panels.get("cb_conversion_premium", close * np.nan)
    bond_premium = panels.get("cb_bond_premium", close * np.nan)
    parity = panels.get("cb_parity", close * np.nan)
    ytm = panels.get("cb_ytm", close * np.nan)
    turnover = panels.get("cb_turnover", close * np.nan)
    iv = panels.get("cb_iv", close * np.nan)

    common_index = close.index
    rows = []
    for dt in common_index:
        price = close.loc[dt]
        bal = balance.reindex(index=[dt]).iloc[0].reindex(price.index)
        amt = amount.reindex(index=[dt]).iloc[0].reindex(price.index)
        premium = conv_premium.reindex(index=[dt]).iloc[0].reindex(price.index)
        bp = bond_premium.reindex(index=[dt]).iloc[0].reindex(price.index)
        par = parity.reindex(index=[dt]).iloc[0].reindex(price.index)
        y = ytm.reindex(index=[dt]).iloc[0].reindex(price.index)
        to = turnover.reindex(index=[dt]).iloc[0].reindex(price.index)
        vol = iv.reindex(index=[dt]).iloc[0].reindex(price.index)
        active = price.notna()
        rows.append(
            {
                "date": dt,
                "cb_count": int(active.sum()),
                "cb_balance_sum": bal[active].sum(min_count=1),
                "cb_amount_sum": amt[active].sum(min_count=1),
                "cb_price_avg": price[active].mean(),
                "cb_price_median": price[active].median(),
                "cb_price_wavg_balance": wavg(price, bal),
                "cb_premium_avg": premium[active].mean(),
                "cb_premium_median": premium[active].median(),
                "cb_premium_wavg_balance": wavg(premium, bal),
                "cb_bond_premium_avg": bp[active].mean(),
                "cb_bond_premium_median": bp[active].median(),
                "cb_parity_median": par[active].median(),
                "cb_ytm_median": y[active].median(),
                "cb_turnover_median": to[active].median(),
                "cb_iv_median": vol[active].median(),
                "low_price_cb_ratio": (price[active] < 110).mean() if active.any() else np.nan,
                "high_price_cb_ratio": (price[active] > 130).mean() if active.any() else np.nan,
                "double_low_proxy_median": (price + premium)[active].median(),
            }
        )

    daily = pd.DataFrame(rows).sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"])
    monthly = daily.groupby(daily["date"].dt.to_period("M")).tail(1).reset_index(drop=True)
    monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly["source"] = "existing_cb_database_tushare_panels"

    daily.to_parquet(PROCESSED / "existing_cb_market_features_daily.parquet", index=False)
    daily.to_csv(PROCESSED / "existing_cb_market_features_daily.csv", index=False, encoding="utf-8-sig")
    monthly.to_parquet(PROCESSED / "existing_cb_market_features_monthly.parquet", index=False)
    monthly.to_csv(PROCESSED / "existing_cb_market_features_monthly.csv", index=False, encoding="utf-8-sig")

    coverage_df = pd.DataFrame(coverage)
    with pd.ExcelWriter(REPORTS / "existing_cb_database_import_report.xlsx", engine="openpyxl") as writer:
        coverage_df.to_excel(writer, sheet_name="source_coverage", index=False)
        monthly.tail(24).to_excel(writer, sheet_name="monthly_tail", index=False)
        monthly.isna().mean().rename("missing_ratio").reset_index().to_excel(writer, sheet_name="missing_ratio", index=False)

    state = STATE_PATH.read_text(encoding="utf-8") if STATE_PATH.exists() else ""
    lines = [
        "# 现有可转债数据库导入报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 日频特征行数：{len(daily)}",
        f"- 月频特征行数：{len(monthly)}",
        f"- 时间范围：{daily['date'].min().date()} 至 {daily['date'].max().date()}",
        f"- 现有转债库状态：{state[:300].replace(chr(10), ' ')}",
        "",
        "## 输出",
        "- data_processed/existing_cb_market_features_daily.parquet",
        "- data_processed/existing_cb_market_features_monthly.parquet",
        "- output/reports/existing_cb_database_import_report.xlsx",
    ]
    (REPORTS / "existing_cb_database_import_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
