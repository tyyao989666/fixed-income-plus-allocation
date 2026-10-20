"""Build quarantined Boundary covariates without changing any prior experiment."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v66_boundary import monthly_boundary, align_completed_months, boundary_features

RAW = ROOT / 'data_raw/boundary_v66'
OUT = ROOT / 'output/reports/boundary_v66'
DATA = ROOT / 'data_processed/boundary_v66'
SOURCE = ROOT / 'data_processed/corporate_action_v58/dataset.npz'
SOURCE_HASH = 'f0c9dcf1839524c678134fe472fabe917e6aa05ce5492f86b34c0d700f4ec613'
NAMES = ['boundary_scaled_dividend_yield', 'boundary_scaled_term_spread',
         'valuation_boundary_10y', 'boundary_x_price_momentum_252d',
         'boundary_x_absolute_price_momentum_252d']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, date_format):
    x = pd.read_csv(RAW / (name+'.csv'))
    x.index = pd.to_datetime(x.pop('日期').astype(str), format=date_format)
    if x.index.has_duplicates:
        raise ValueError('Duplicate source dates: '+name)
    return x.sort_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if sha(SOURCE) != SOURCE_HASH:
        raise ValueError('Corrected 38-asset dataset identity changed')
    dividend = load('dividend_history', '%Y%m%d')['股息率'].apply(pd.to_numeric, errors='coerce')
    rates = load('treasury_history', '%Y-%m-%d').apply(pd.to_numeric, errors='coerce')
    close = load('equity_close_history', '%Y%m%d')['收盘价（不前推）'].apply(pd.to_numeric, errors='coerce').dropna()
    if (close <= 0).any() or (close.index.dayofweek >= 5).any():
        raise ValueError('Invalid index closing-price calendar')
    spread = rates['中债国债到期收益率:10年']-rates['中债国债到期收益率:1年']
    month_end_dates = close.groupby(close.index.to_period('M')).apply(lambda s: s.index[-1])
    # Drop the incomplete latest month; never interpret its last observation as month end.
    month_end_dates = month_end_dates[month_end_dates.index < close.index[-1].to_period('M')]
    monthly = pd.DataFrame({'dividend_yield': dividend.reindex(month_end_dates.values).to_numpy(),
                            'term_spread': spread.reindex(month_end_dates.values).to_numpy()},
                           index=month_end_dates.index)
    monthly['source_date'] = month_end_dates.values
    for h in [120, 240]:
        b = monthly_boundary(monthly, history_months=h)
        b.to_csv(OUT / f'monthly_boundary_{h}m.csv', encoding='utf-8-sig')
    monthly.to_csv(OUT / 'monthly_inputs.csv', encoding='utf-8-sig')
    with np.load(SOURCE, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    dates = pd.DatetimeIndex(pd.to_datetime(a['dates']))
    boundary = monthly_boundary(monthly)
    daily = align_completed_months(boundary, dates, certified=False)
    momentum = close.pct_change(252, fill_method=None).reindex(dates).to_numpy()
    values, research_mask, strict_mask = boundary_features(daily, momentum, certified=False)
    daily['price_momentum_252d'] = momentum
    daily.to_csv(OUT / 'daily_boundary_state.csv', encoding='utf-8-sig', index_label='date')
    # Raw exploratory covariates remain separate from the strict model input.
    np.savez_compressed(DATA / 'research_boundary_features.npz', dates=a['dates'],
        values=values, research_masks=research_mask, strict_masks=strict_mask,
        feature_names=np.array(NAMES))
    original_width = a['macro_values'].shape[-1]
    a['macro_values'] = np.concatenate([a['macro_values'], np.where(strict_mask, values, 0).astype(a['macro_values'].dtype)], axis=-1)
    a['macro_masks'] = np.concatenate([a['macro_masks'], strict_mask.astype(a['macro_masks'].dtype)], axis=-1)
    np.savez_compressed(DATA / 'dataset_strict_masked.npz', **a)
    with (OUT / 'market_boundary_cards.jsonl').open('w', encoding='utf-8') as f:
        for i, date in enumerate(dates):
            f.write(json.dumps(dict(date=str(date.date()), observation_month=daily.observation_month.iloc[i],
                equity_valuation_boundary=dict(value=float(values[i, 2]) if research_mask[i, 2] else None,
                    scaled_dividend_yield=float(values[i, 0]) if research_mask[i, 0] else None,
                    scaled_term_spread=float(values[i, 1]) if research_mask[i, 1] else None,
                    strict_eligible=False, status='research_only_unverified_dividend_definition_and_vintage',
                    scope='CSI300_equity_context_not_all_asset_reversal', unit='dimensionless'),
                existing_nine_grid=int(a['regime_index'][i]), asset_universe_count=38), ensure_ascii=False)+'\n')
    # Verify the feature pipeline is prefix-causal, beyond synthetic unit tests.
    prefix_checks = []
    for length in [135, 180, len(monthly)-1]:
        pd.testing.assert_frame_equal(boundary.iloc[:length], monthly_boundary(monthly.iloc[:length]))
        prefix_checks.append(length)
    dates_match = dates.isin(close.index)
    if not dates_match.all():
        raise ValueError('New index calendar misses corrected-dataset trading dates')
    source_months = pd.PeriodIndex(daily.observation_month, freq='M')
    assert (source_months < dates.to_period('M')).all()
    assert not strict_mask.any()
    assert a['module_index'].shape == (38,)
    coverage = []
    for year in sorted(set(dates.year)):
        ix = dates.year == year
        coverage.append(dict(year=int(year), days=int(ix.sum()), research_boundary_days=int(research_mask[ix, 2].sum()),
                             strict_boundary_days=int(strict_mask[ix, 2].sum())))
    pd.DataFrame(coverage).to_csv(OUT / 'coverage_by_year.csv', index=False)
    identities = {str(p.relative_to(ROOT)): sha(p) for p in [SOURCE, Path(__file__),
        ROOT / 'src/v66_boundary.py', ROOT / 'docs/v66_boundary_protocol.md',
        *[RAW / (name+'.csv') for name in ['dividend_history', 'treasury_history', 'equity_close_history']]]}
    status = dict(paper='https://doi.org/10.1111/fima.70055', paper_verified=True,
        source_hashes=identities, source_dataset_preserved=True, asset_count=38, nine_grid_preserved=True,
        original_macro_width=original_width, augmented_macro_width=a['macro_values'].shape[-1],
        first_research_month=str(boundary.boundary.first_valid_index()),
        history_months=120, smoothing_months=12, range_on='smoothed_monthly_series',
        term_spread='ChinaBond_10y_minus_1y_percentage_points_China_adaptation',
        dividend_yield_unit='percent_vendor_definition_unverified',
        monthly_lag='previous_completed_calendar_month', prefix_causality_checks=prefix_checks,
        daily_rows=len(dates), research_boundary_days=int(research_mask[:, 2].sum()), strict_boundary_days=0,
        source_calendar_covers_dataset=bool(dates_match.all()),
        strict_blockers=['Dividend TTM/constituent definition not established',
                         'Historical release/revision vintages not certified'],
        training_run=False, portfolio_backtest_run=False, promotion=False,
        output_hashes={str(p.relative_to(ROOT)):sha(p) for p in DATA.glob('*.npz')})
    (OUT / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in status.items() if k not in ['source_hashes', 'output_hashes']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
