"""Security-level missing-holding evidence queue; never invent an exit return."""
import json
import hashlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.audit_v67_external_boundary import EXTERNAL, CASH_PATH, positions

OUT = ROOT / 'output/reports/cashflow_repair_v68'


def parse_date(value):
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    return pd.to_datetime(str(value), format='%Y%m%d', errors='coerce')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sh = pd.read_parquet(CASH_PATH)
    sh['date'] = pd.to_datetime(sh.date.astype(str))
    cash = sh.sort_values('date').set_index('date')['1y'].resample('ME').last()/1200
    basic_path = ROOT / 'data_raw/tushare/cb_basic_all.parquet'
    basic = pd.read_parquet(basic_path).drop_duplicates('ts_code').set_index('ts_code')
    evidence_path = ROOT / 'data_processed/cb_execution_v54/reviewed_terminal_event_with_evidence.parquet'
    events = pd.read_parquet(evidence_path) if evidence_path.exists() else pd.DataFrame()
    paths = [CASH_PATH, basic_path, Path(__file__)]
    if evidence_path.exists():
        paths.append(evidence_path)
    details, price_audits, signal_audits = [], [], []
    for asset, file in [('stocks','stocks_monthly_pti.parquet'),('cb','cb_monthly_pti.parquet')]:
        path = EXTERNAL / 'work/individual_point_in_time_data' / file
        paths.append(path)
        d = pd.read_parquet(path)
        if asset == 'stocks':
            code = d.ts_code.astype(str)
            d = d[((code.str.endswith('.SH') & code.str[:3].isin(['600','601','603','605','688'])) |
                   (code.str.endswith('.SZ') & code.str[:3].isin(['000','001','002','003','300','301'])) |
                   code.str.endswith('.BJ'))].copy()
        d['date'] = pd.to_datetime(d.trade_date).dt.to_period('M').dt.to_timestamp('M')
        d = d[d.date <= pd.Timestamp('2025-03-31')].copy()
        d['close'] = pd.to_numeric(d.close, errors='coerce')
        d['amount'] = pd.to_numeric(d.amount, errors='coerce')
        close = d.pivot(index='date', columns='ts_code', values='close').sort_index()
        close = close.reindex(pd.date_range(close.index.min(), close.index.max(), freq='ME'))
        amount = d.pivot(index='date', columns='ts_code', values='amount').reindex_like(close)
        idx = close.index.intersection(cash.dropna().index)
        close, amount, c = close.loc[idx], amount.loc[idx], cash.loc[idx]
        ret = close.pct_change(fill_method=None).where(close.gt(0))
        base, _ = positions(ret.sub(c, axis=0))
        eligible = amount.gt(0) & close.gt(0) & base.notna()
        weights = base.where(eligible,0).div(eligible.sum(1).replace(0,np.nan),axis=0).fillna(0)
        previous = weights.shift(1).fillna(0)
        # Fixed common start of the original 24-month Boundary experiment.
        assessment = idx >= pd.Timestamp('2021-04-30')
        unknown = ret.isna() & previous.gt(1e-12)
        for t, j in np.argwhere(unknown.to_numpy() & assessment[:,None]):
            date, code = idx[t], close.columns[j]
            future = close[code].iloc[t+1:].dropna()
            later_date = future.index[0] if len(future) else pd.NaT
            meta = basic.loc[code] if asset=='cb' and code in basic.index else None
            delist = parse_date(meta['delist_date']) if meta is not None else pd.NaT
            maturity = parse_date(meta['maturity_date']) if meta is not None else pd.NaT
            present_quote = pd.notna(close.iat[t,j])
            if present_quote:
                kind = 'prior_month_quote_gap'
            elif pd.notna(delist) and idx[t-1] < delist <= date:
                kind = 'terminal_in_month_metadata_only'
            elif pd.notna(delist) and delist <= idx[t-1]:
                kind = 'already_delisted_but_prior_quote_present'
            elif pd.notna(later_date):
                kind = 'quote_reappears_later_unknown_suspension_or_gap'
            else:
                kind = 'no_later_quote_unknown_terminal_or_gap'
            reviewed = events[events.ts_code.eq(code)] if len(events) else pd.DataFrame()
            matched = reviewed[pd.to_datetime(reviewed.pay_date).dt.to_period('M').eq(date.to_period('M'))] if len(reviewed) else reviewed
            details.append(dict(asset=asset, ts_code=code, signal_date=str(idx[t-1].date()),
                return_month=str(date.date()), held_weight=float(previous.iat[t,j]),
                previous_close=float(close.iat[t-1,j]) if pd.notna(close.iat[t-1,j]) else None,
                current_close=float(close.iat[t,j]) if present_quote else None,
                evidence_category=kind, later_quote_date=str(later_date.date()) if pd.notna(later_date) else None,
                delist_date=str(delist.date()) if pd.notna(delist) else None,
                maturity_date=str(maturity.date()) if pd.notna(maturity) else None,
                name=meta.bond_short_name if meta is not None else None,
                matched_reviewed_terminal_evidence=bool(len(matched)),
                event_applied_to_returns=False, historical_eligibility_rewritten=False,
                needs='terminal_cashflow_or_suspension_and_valuation_evidence'))
        # Monthly close-to-close jumps are alerts, not proof of a split or default.
        for t,j in np.argwhere(((ret <= -.30) | (ret >= .80)).to_numpy()):
            price_audits.append(dict(asset=asset, ts_code=close.columns[j], date=str(idx[t].date()),
                raw_monthly_price_return=float(ret.iat[t,j]), status='alert_not_verified_corporate_action',
                repair_applied=False))
        signal_audits.append(dict(asset=asset, source_api='monthly' if asset=='stocks' else 'cb_daily_month_end_snapshots',
            raw_close_only=True, corporate_action_adjustment_verified=False, coupons_verified=False,
            first_month=str(idx[0].date()), last_month=str(idx[-1].date()),
            missing_held_security_months=int((unknown.to_numpy() & assessment[:,None]).sum()),
            documented_scope='base_TSM_only_fixed_Apr2021_start_no_Boundary_parameter_selection'))
    detail = pd.DataFrame(details)
    detail.to_csv(OUT / 'security_month_missing_ledger.csv', index=False, encoding='utf-8-sig')
    queue = detail.groupby(['asset','ts_code'], as_index=False).agg(
        missing_months=('return_month','size'), first_missing=('return_month','min'),
        last_missing=('return_month','max'), summed_weight_months=('held_weight','sum'),
        max_held_weight=('held_weight','max'), reviewed_event_matches=('matched_reviewed_terminal_evidence','sum'))
    queue = queue.sort_values(['asset','summed_weight_months'], ascending=[True,False])
    queue.to_csv(OUT / 'repair_priority_by_security.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(price_audits).to_csv(OUT / 'raw_price_jump_alerts.csv', index=False)
    counts = detail.groupby(['asset','evidence_category'],as_index=False).agg(
        security_months=('ts_code','size'), securities=('ts_code','nunique'), weight_months=('held_weight','sum'))
    counts.to_csv(OUT / 'evidence_category_summary.csv', index=False)
    pd.DataFrame(signal_audits).to_csv(OUT / 'source_semantics_audit.csv', index=False)
    status = dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        no_network=True, no_training=True, historical_returns_modified=False,
        classification_uses_later_quotes_for_forensic_audit_only=True,
        later_metadata_never_used_to_rewrite_historical_universe=True,
        formal_performance_certified=False, verified_settlements_automatically_applied=0,
        units='weight_months are sums across dependent months, NOT NAV loss or independent sample counts',
        scope='base_TSM_missing_holdings; not all flip/soft_flip holdings',
        queue_securities=len(queue), ledger_rows=len(detail), price_jump_alerts=len(price_audits))
    (OUT / 'status.json').write_text(json.dumps(status, indent=2),encoding='utf-8')
    print(counts.to_string(index=False))
    print(queue.groupby('asset',sort=False).head(5).to_string(index=False))


if __name__=='__main__':
    main()
