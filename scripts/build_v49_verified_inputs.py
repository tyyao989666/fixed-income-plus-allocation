"""Build isolated PIT candidates and explicit blockers, never splice histories."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.stage_20261014_market_data import tables
from src.v45_forecast import write_json
from src.v49_point_in_time import cb_disclosure_events, align_disclosed_balances, cb_bucket_weights, realized_bucket_return, dividend_cash_per_unit

BASE = ROOT / "data_raw/refresh_20261014_wind_verified"
SOURCE = BASE / "supplement_v49"
OUT = ROOT / "data_processed/v49_review_candidates"
REPORTS = ROOT / "output/reports/v49_project_review"


def exact_macro(name, expected_name, expected_source):
    response = json.loads((SOURCE / "wind" / f"{name}.json").read_text(encoding="utf-8"))["response"]
    matches = []
    for frame, _ in tables(response):
        if not {"name", "source", "unit"}.issubset(frame):
            continue
        selected = frame.loc[frame.name.eq(expected_name) & frame.source.eq(expected_source) & frame.unit.eq("%")]
        if not selected.empty:
            matches.append(selected.copy())
    if len(matches) != 1:
        raise ValueError(f"Macro identity ambiguous or unavailable: {name}")
    frame = matches[0]
    frame["period_date"] = pd.to_datetime(frame.period_date.astype(str), errors="raise")
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    frame["release_verified"] = False
    frame["training_allowed"] = False
    return frame


def ifind_curve(name, expected):
    raw = json.loads((SOURCE / "ifind" / f"{name}.json").read_text(encoding="utf-8"))
    result = raw["response"]["data"]["result"]
    payload = json.loads(result["content"][0]["text"])["data"]["datas"]
    matches = []
    for value in payload:
        table = value["data"]
        if expected not in table.get("columns", []):
            continue
        attrs = table.get("attrs", {}).get(expected, {})
        if attrs.get("unit") != "%" or attrs.get("data_source") != "中国债券信息网":
            raise ValueError("Unexpected ChinaBond source or units")
        frame = pd.DataFrame(table["data"], columns=table["columns"])
        matches.append(pd.DataFrame({"date":pd.to_datetime(frame.iloc[:,0], errors="raise"),
                                     "yield_percent":pd.to_numeric(frame[expected], errors="raise")}))
    if len(matches) != 1 or matches[0].date.duplicated().any():
        raise ValueError("Ambiguous curve table")
    frame = matches[0].sort_values("date")
    if frame.date.max() > pd.Timestamp("2026-10-14"):
        raise ValueError("Curve exceeded cutoff")
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_parquet(BASE / "tushare_cb/calendar.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(sorted(calendar.cal_date.astype(str)), format="%Y%m%d"))
    panels = pd.concat([pd.read_parquet(p) for p in (BASE / "tushare_cb").glob("cb_daily_*.parquet")], ignore_index=True)
    panels["date"] = pd.to_datetime(panels.trade_date, format="%Y%m%d")
    if panels.duplicated(["date", "ts_code"]).any():
        raise ValueError("Duplicate CB market observations")
    close = panels.pivot(index="date", columns="ts_code", values="close").reindex(dates)
    # Tushare's explicitly observed previous close permits validating the
    # individual daily return even when the previous day's quote was absent.
    panels["observed_return"] = panels.close/panels.pre_close-1
    returns = panels.pivot(index="date", columns="ts_code", values="observed_return").reindex(dates)
    disclosures = pd.read_parquet(SOURCE / "candidates/cb_share_dated_disclosures.parquet")
    events = cb_disclosure_events(disclosures, dates, "2026-10-14")
    known = align_disclosed_balances(events, dates, list(close), max_age_days=120)
    events.to_parquet(OUT / "cb_dated_events.parquet", index=False)
    known["balance"].to_parquet(OUT / "cb_last_published_balance.parquet")
    known["age_days"].to_parquet(OUT / "cb_balance_announcement_age.parquet")
    buckets, ledger, missing_positions = [], [], []
    for i in range(1, len(dates)):
        previous, day = dates[i-1], dates[i]
        weights = cb_bucket_weights(close.loc[previous], known["balance"].loc[previous], known["issue_size"].loc[previous])
        for name, detail in weights.items():
            w = detail["weights"]
            value = realized_bucket_return(w, returns.loc[day])
            for code in w.index[returns.loc[day].reindex(w.index).isna()]:
                missing_positions.append(dict(date=day, signal_date=previous, asset_code=name,
                                              ts_code=code, precommitted_weight=float(w[code])))
            buckets.append(dict(date=day, signal_date=previous, asset_code=name, return_1d=value))
            ledger.append(dict(date=day, signal_date=previous, asset_code=name, member_count=len(w),
                max_weight=float(w.max()), weight_sum=float(w.sum()), fallback_level=detail["fallback_level"],
                missing_held_returns=int(returns.loc[day].reindex(w.index).isna().sum()),
                membership_sha256=hashlib.sha256(w.sort_index().to_json().encode()).hexdigest()))
    bucket = pd.DataFrame(buckets)
    log = pd.DataFrame(ledger)
    bucket.to_parquet(OUT / "cb_12_buckets_causal_candidates.parquet", index=False)
    log.to_csv(REPORTS / "cb_bucket_formation_audit.csv", index=False)
    pd.DataFrame(missing_positions).to_csv(REPORTS / "cb_missing_held_components.csv", index=False)
    reits, reits_cash = {}, {}
    for path in (SOURCE / "tushare").glob("reits_unadjusted_*.parquet"):
        code = path.stem.removeprefix("reits_unadjusted_")
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame.trade_date, format="%Y%m%d")
        if frame.date.duplicated().any() or not frame.ts_code.eq(code).all():
            raise ValueError("Ambiguous unadjusted REIT source")
        price = frame.set_index("date").close.reindex(dates)
        dividend = pd.read_parquet(SOURCE / "tushare" / f"div_{code}.parquet")
        cash = dividend_cash_per_unit(dividend, dates, code)
        reits_cash[code] = cash
        reits[code] = (price+cash)/price.shift(1)-1
    reit_matrix = pd.DataFrame(reits, index=dates)
    if len(reit_matrix.columns) != 9:
        raise ValueError("Expected all nine fixed REIT components")
    reit_matrix.to_parquet(OUT / "reits_component_total_return_candidates.parquet")
    pd.DataFrame(reits_cash).to_parquet(OUT / "reits_cash_dividends_once.parquet")
    pd.DataFrame({"date":dates, "asset_code":"alt_reits", "return_1d":reit_matrix.mean(axis=1, skipna=False),
                  "convention":"fixed9_daily_equal_weight_gross_exdate_reinvestment_research_proxy"}).to_parquet(
                      OUT / "reits_fixed9_total_return_candidate.parquet", index=False)
    macro_inventory = []
    for key, name, source in [("cpi_exact", "中国:CPI:当月同比", "国家统计局"),
                              ("ppi_exact", "中国:PPI:当月同比", "国家统计局"),
                              ("pmi_exact", "中国:制造业PMI", "国家统计局"),
                              ("dr007_exact", "DR007", "中国货币网"),
                              ("r007_exact", "中国:银行间质押式回购加权利率:7天", "中国货币网")]:
        frame = exact_macro(key, name, source)
        frame.to_parquet(OUT / f"{key}_identity_verified_release_pending.parquet", index=False)
        macro_inventory.append(dict(field=key, identity_verified=True, release_verified=False,
                                    observations=int(frame.value.notna().sum()), training_allowed=False))
    curves = []
    for key, expected in [("treasury30", "中债国债到期收益率:30年"),
                          ("treasury10", "中债国债到期收益率:10年"),
                          ("creditAAA3", "中债中短期票据到期收益率(AAA):3年")]:
        try:
            frame = ifind_curve(key, expected)
            frame.to_parquet(OUT / f"{key}_source_verified.parquet", index=False)
            aligned = frame.set_index("date").yield_percent.reindex(dates)
            curves.append(dict(field=key, source_verified=True, calendar_missing=int(aligned.isna().sum()),
                               note="publish timestamps and same-source history need separate approval"))
        except (ValueError, KeyError, TypeError) as exc:
            curves.append(dict(field=key, source_verified=False, error=str(exc)))
    reasons = ["full_history_CB_constituent_rebuild_required_before_old_metrics_can_be_revalidated",
        "macro_actual_release_and_vintage_unverified_not_in_new_training_inputs",
        "REIT_research_total_return_candidate_requires_historical_rebuild_and_reinvestment_cost_review",
        "USD_curve_FX_and_domestic_visibility_contract_unverified",
        "30Y_model_token_is_duration_curve_proxy_not_10Yplus_wealth_index",
        "CB_bucket_internal_reconstitution_cost_not_yet_modelled_separately_from_asset_allocation_cost"]
    if log.missing_held_returns.gt(0).any():
        reasons.append("CB_missing_held_next_returns_kept_missing_terminal_payoffs_needed")
    summary = dict(training_ready=False, formal_replacement=False, blockers=reasons,
        deployment_blockers=["masked_encoder_requires_fresh_training_and_independent_forecast_gate",
                             "three_product_portfolio_and_PPO_promotion_not_passed"],
        cutoff="2026-10-14", cb_event_rows=len(events), cb_event_codes=events.ts_code.nunique(),
        cb_bucket_rows=len(bucket), cb_buckets=bucket.asset_code.nunique(),
        cb_missing_return_rows=int(bucket.return_1d.isna().sum()), max_single_bond_weight=float(log.max_weight.max()),
        reit_components=len(reit_matrix.columns), reit_days_with_all_returns=int(reit_matrix.notna().all(axis=1).sum()),
        cb_weight_sums_valid=bool(np.allclose(log.weight_sum, 1, atol=1e-10)),
        macro=macro_inventory, curves=curves,
        historical_splicing="PROHIBITED; repaired weights and observation semantics differ from old frozen history",
        candidate_directory=str(OUT))
    write_json(REPORTS / "data_readiness.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
