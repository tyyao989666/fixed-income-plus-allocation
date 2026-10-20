"""Read-only verification of the saved v43 research snapshot."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/reports'
PREFIX = 'hierarchical_asset_itransformer_v43_clean_calendar'


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def config(name):
    value = yaml.safe_load((ROOT / 'config' / name).read_text(encoding='utf-8'))
    parent = value.get('base_config')
    return merge(config(parent), value) if parent else value


def main():
    cfg = config(PREFIX + '.yaml')
    arrays = np.load(ROOT / cfg['outputs']['dataset_npz'])
    dates = pd.DatetimeIndex(pd.to_datetime(arrays['dates']))
    candidates = arrays['sample_indices']
    metadata = json.loads((ROOT / cfg['outputs']['metadata']).read_text(encoding='utf-8'))
    positions = [metadata['horizons_days'].index(h) for h in cfg['walk_forward']['split_horizons_days']]
    rows = []
    for year in range(cfg['walk_forward']['first_prediction_year'], dates.max().year + 1):
        cutoff = pd.Timestamp(year, 1, 1)
        eligible = candidates[dates[candidates] < cutoff][::max(cfg['data'].get('train_stride_days', 1), 1)]
        labels = arrays['label_end'][np.ix_(eligible, positions)]
        mature = ((labels > np.iinfo(np.int64).min) & (labels < cutoff.value)).all(axis=1)
        eligible = eligible[mature]
        count = min(max(cfg['walk_forward']['validation_window_days'], cfg['walk_forward']['minimum_validation_samples']), len(eligible) - cfg['walk_forward']['minimum_train_samples'])
        validation = eligible[-count:]
        preliminary = eligible[:-count]
        labels = arrays['label_end'][np.ix_(preliminary, positions)]
        keep = ((labels <= np.iinfo(np.int64).min) | (labels < dates[validation[0]].value)).all(axis=1)
        train = preliminary[keep]
        test = candidates[dates[candidates].year == year]
        rows.append(dict(year=year, train_start=str(dates[train[0]].date()), train_end=str(dates[train[-1]].date()), train_n=len(train), purge_n=len(preliminary)-len(train), val_start=str(dates[validation[0]].date()), val_end=str(dates[validation[-1]].date()), val_n=len(validation), test_start=str(dates[test[0]].date()), test_end=str(dates[test[-1]].date()), test_n=len(test)))
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / 'v43_review_20261014_fold_boundaries.csv', index=False, encoding='utf-8-sig')
    metrics = []
    checks = []
    for profile in ['low_vol', 'balanced', 'high_vol']:
        daily = pd.read_csv(OUT / f'{PREFIX}_{profile}_daily.csv', parse_dates=['date','signal_date'])
        sample = daily.loc[daily.date.ge('2024-01-01')]
        for strategy in ['strategy','formal','mechanical']:
            r = sample[f'{strategy}_return'].to_numpy()
            vol = r.std(ddof=0)*np.sqrt(252)
            annual = np.prod(1+r)**(252/len(r))-1
            metrics.append(dict(profile=profile,strategy=strategy,days=len(r),annual_return=annual,annual_vol=vol,reported_cagr_over_vol=annual/vol,arithmetic_sharpe_rf0=r.mean()/r.std(ddof=1)*np.sqrt(252),start=str(sample.date.min().date()),end=str(sample.date.max().date())))
        checks.append(dict(profile=profile,check='daily_return_finite',passed=bool(np.isfinite(daily.strategy_return).all())))
        checks.append(dict(profile=profile,check='signal_before_execution',passed=bool((daily.signal_date < daily.date).all())))
        checks.append(dict(profile=profile,check='cost_equals_5bp_l1',passed=bool(np.allclose(daily.transaction_cost, daily.turnover*0.0005))))
        for kind in ['weights','targets']:
            weight = pd.read_parquet(ROOT / f'data_processed/{PREFIX}_{profile}_{kind}_daily.parquet')
            sums = weight.groupby('date').weight.sum()
            checks.append(dict(profile=profile,check=f'{kind}_sum_one',passed=bool(np.allclose(sums,1,atol=1e-6))))
            checks.append(dict(profile=profile,check=f'{kind}_no_negative',passed=bool(weight.weight.ge(-1e-10).all())))
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / 'v43_review_20261014_recomputed_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT / 'v43_review_20261014_saved_output_checks.csv',index=False,encoding='utf-8-sig')
    raw = pd.read_parquet(ROOT / 'data_processed/asset_returns_daily.parquet')
    coverage = raw.groupby('asset_code').agg(start=('date','min'),end=('date','max'),rows=('date','size'),missing_returns=('return_1d',lambda x:x.isna().sum()),source=('data_source',lambda x:' | '.join(sorted(x.dropna().unique()))))
    coverage.to_csv(OUT / 'v43_review_20261014_asset_coverage.csv',encoding='utf-8-sig')
    print(folds.to_string(index=False))
    print(metric_frame.to_string(index=False))
    print(pd.DataFrame(checks).to_string(index=False))
    print('asset_count',len(coverage),'raw_end',raw.date.max(),'weekend_rows',int((pd.to_datetime(raw.date).dt.dayofweek>=5).sum()))
    print(coverage.loc[[x for x in coverage.index if str(x).startswith('rate_')]].to_string())
    print('controller_config',json.dumps(cfg['controller'],ensure_ascii=False))
    print('model_config',json.dumps({k:cfg['model'].get(k) for k in ['d_model','n_heads','num_layers','e_layers','dropout','batch_size']},ensure_ascii=False))
    comparison = pd.DataFrame({'date':dates,'a2_model_proxy_return':arrays['returns'][:,metadata['assets'].index('rate_30y_curve_proxy')]})
    wealth = raw.loc[raw.asset_code.eq('rate_30y'),['date','return_1d']].rename(columns={'return_1d':'a0_a1_wealth_return'})
    comparison = comparison.merge(wealth,on='date',how='left')
    comparison.to_csv(OUT / 'v43_review_20261014_30y_source_comparison.csv',index=False,encoding='utf-8-sig')
    print('30y_returns_max_difference',float((comparison.a2_model_proxy_return-comparison.a0_a1_wealth_return).abs().max()))
    print('30y_returns_correlation',comparison[['a2_model_proxy_return','a0_a1_wealth_return']].corr().iloc[0,1])


if __name__ == '__main__':
    main()
