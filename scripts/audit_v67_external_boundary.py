"""Fixed representative old-rule accounting audit; never selects a best setting."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v66_boundary import monthly_boundary
from src.v67_monthly_accounting import monthly_ledger

EXTERNAL = Path('C:/Users/lenovo/Documents/Codex/2026-08-31/https-quantocracy-com-https-quantocracy-com')
CASH_PATH = Path('C:/Users/lenovo/Documents/Codex/2026-05-21/ai-ai-ai-ai-agent-aifin/fixed_income_plus_allocation/data_raw/tushare/shibor_shibor.parquet')
OUT = ROOT / 'output/reports/external_boundary_audit_v67'


def metrics(x):
    x = x.dropna()
    nav = (1+x).cumprod()
    # Include initial wealth 1 in the high-water mark.
    dd = nav/np.maximum(nav.cummax(), 1)-1
    return dict(months=len(x), cagr=float(nav.iloc[-1]**(12/len(x))-1),
        sharpe_rf0=float(x.mean()/x.std()*np.sqrt(12)), max_drawdown=float(dd.min()))


def positions(excess):
    votes = []
    for lb in [1,3,6,9,12]:
        mom = (1+excess).rolling(lb, min_periods=lb).apply(np.prod, raw=True)-1
        sig = (mom > 0).astype(float).where(mom.notna())
        for hold in [1,3,6,9,12]:
            votes.append(sig.rolling(hold, min_periods=1).mean())
    return sum(votes)/25, (1+excess).rolling(12, min_periods=12).apply(np.prod, raw=True)-1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sh = pd.read_parquet(CASH_PATH)
    sh['date'] = pd.to_datetime(sh.date.astype(str))
    cash = sh.sort_values('date').set_index('date')['1y'].resample('ME').last()/1200
    raw = json.loads((EXTERNAL / 'work/boundaries_replication/hs300_valuation.json').read_text(encoding='utf-8'))
    val = pd.DataFrame(raw['rows'], columns=raw['columns'])
    val['date'] = pd.to_datetime(val.iloc[:, 2]).dt.to_period('M')
    dy = pd.Series(pd.to_numeric(val.iloc[:, 4], errors='coerce').to_numpy(), index=val.date).groupby(level=0).last()
    curve = pd.DataFrame(json.loads((EXTERNAL / 'work/boundaries_replication/cn_govt_10y_daily.json').read_text(encoding='utf-8')))
    curve = curve[curve.date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}')].copy()
    curve['date'] = pd.to_datetime(curve.date)
    y10 = pd.to_numeric(curve.set_index('date').y10, errors='coerce').resample('ME').last()
    spread = y10-sh.sort_values('date').set_index('date')['1y'].resample('ME').last()
    monthly = pd.concat([dy.rename('dividend_yield'), pd.Series(spread.to_numpy(), index=spread.index.to_period('M'), name='term_spread')], axis=1).sort_index()
    rows, missing_rows, coverage, paths = [], [], [], [CASH_PATH]
    for name, file, cost in [('stocks','stocks_monthly_pti.parquet',20), ('cb','cb_monthly_pti.parquet',30)]:
        path = EXTERNAL / 'work/individual_point_in_time_data' / file
        paths.append(path)
        d = pd.read_parquet(path)
        if name == 'stocks':
            code = d.ts_code.astype(str)
            keep = ((code.str.endswith('.SH') & code.str[:3].isin(['600','601','603','605','688'])) |
                    (code.str.endswith('.SZ') & code.str[:3].isin(['000','001','002','003','300','301'])) | code.str.endswith('.BJ'))
            d = d[keep].copy()
        d['date'] = pd.to_datetime(d.trade_date).dt.to_period('M').dt.to_timestamp('M')
        d = d[d.date <= pd.Timestamp('2025-03-31')]
        d['close'] = pd.to_numeric(d.close, errors='coerce')
        d['amount'] = pd.to_numeric(d.amount, errors='coerce')
        close = d.pivot(index='date', columns='ts_code', values='close').sort_index()
        close = close.reindex(pd.date_range(close.index.min(), close.index.max(), freq='ME'))
        amount = d.pivot(index='date', columns='ts_code', values='amount').reindex_like(close)
        idx = close.index.intersection(cash.dropna().index)
        close, amount, c = close.loc[idx], amount.loc[idx], cash.loc[idx]
        returns = dict(legacy_ffill=close.ffill().pct_change(fill_method=None).where(close.gt(0)),
                       no_fill=close.pct_change(fill_method=None).where(close.gt(0)))
        for basis, ret in returns.items():
            excess = ret.sub(c, axis=0)
            base, past12 = positions(excess)
            for window in [24,36]:
                b = monthly_boundary(monthly, history_months=window).boundary
                b.index = b.index.to_timestamp('M')
                b = b.reindex(idx)
                for policy in ['base','flip','soft_flip']:
                    p = base.copy()
                    gate = past12.abs().ge(.10) & b.ge(1).to_numpy()[:, None]
                    if policy != 'base':
                        p = p.where(~gate, 1-base if policy == 'flip' else .5)
                    eligible = amount.gt(0) & close.gt(0) & p.notna()
                    count = eligible.sum(1).replace(0, np.nan)
                    target = p.where(eligible, 0).div(count, axis=0).fillna(0)
                    valid = b.shift(1).notna()
                    prior = target.shift(1).fillna(0)
                    legacy_gross = c+(prior*excess.fillna(0)).sum(1)
                    legacy_fee = (target-target.shift(1).fillna(0)).abs().sum(1).shift(1).fillna(0)/2*cost/10000
                    ledger = monthly_ledger(ret, target, c)
                    correction = ledger.gross_return-ledger.risky_traded_notional*cost/10000
                    for accounting, result in [('old_half_L1',legacy_gross-legacy_fee), ('drift_full_traded',correction)]:
                        for event_rule in ['all', 'exclude_2024_09']:
                            use = valid.copy()
                            if event_rule != 'all':
                                use &= idx != pd.Timestamp('2024-09-30')
                            if not use.any():
                                continue
                            rows.append(dict(asset=name, return_basis=basis, window=window, policy=policy,
                                accounting=accounting, event_rule=event_rule, cost_bps=cost,
                                **metrics(result[use]), unknown_months=int((ledger.unknown_held_weight[use]>0).sum()),
                                max_unknown_weight=float(ledger.unknown_held_weight[use].max()), certified=False))
                    for date, row in ledger.loc[valid & (ledger.unknown_held_weight > 0)].iterrows():
                        missing_rows.append(dict(asset=name, return_basis=basis, window=window, policy=policy,
                            date=str(date.date()), **row.to_dict()))
                coverage.append(dict(asset=name, return_basis=basis, window=window,
                    usable_months=int(valid.sum()), total_names=len(close.columns),
                    mean_signal_eligible=float((amount.gt(0)&close.gt(0)&base.notna()).sum(1)[valid].mean())))
    results = pd.DataFrame(rows)
    keys = ['asset','return_basis','window','accounting','event_rule']
    baseline = results[results.policy=='base'][keys+['cagr','sharpe_rf0']].rename(columns={'cagr':'base_cagr','sharpe_rf0':'base_sharpe'})
    results = results.merge(baseline, on=keys, validate='many_to_one')
    results['delta_cagr'] = results.cagr-results.base_cagr
    results['delta_sharpe'] = results.sharpe_rf0-results.base_sharpe
    results.to_csv(OUT / 'fixed_accounting_comparison.csv', index=False)
    pd.DataFrame(missing_rows).to_csv(OUT / 'unidentified_held_returns.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'signal_coverage.csv', index=False)
    paths += [EXTERNAL / 'work/run_individual_boundary_tsmom.py', EXTERNAL / 'work/run_boundaries_replication_v2.py',
        EXTERNAL / 'work/boundaries_replication/hs300_valuation.json', EXTERNAL / 'work/boundaries_replication/cn_govt_10y_daily.json']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    status = dict(source_hashes=hashes, parameter_search=False, threshold_fixed=1, windows=[24,36],
        missing_scenario='missing held risky return assumed 0; flagged, NOT certified or conservative guarantee',
        cost_convention='fee_bps times absolute risky-asset traded notional; drift at gross NAV; fee financing drift omitted',
        price_basis='raw close; corporate actions, dividends and redemption proceeds NOT certified',
        boundary='old 10Y minus 1Y Shibor; NOT V66 10Y minus 1Y Treasury; 24/36 month adaptation',
        formal_portfolio=False, promotion=False, all_settings_reported=True)
    (OUT / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    view = results[(results.return_basis=='no_fill') & (results.accounting=='drift_full_traded') & (results.policy!='base')]
    print(view[['asset','window','policy','event_rule','delta_cagr','delta_sharpe','unknown_months','max_unknown_weight']].to_string(index=False))


if __name__ == '__main__':
    main()
