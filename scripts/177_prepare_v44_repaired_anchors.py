"""Rebuild mechanical monthly anchors from the exact frozen model return panel."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.portfolio_accounting import ACCOUNTING_VERSION, accounting_return_panel
from src.v1_strategy import V1Paths, load_config, build_asset_weights


def main():
    panel, modules, tradable, gaps = accounting_return_panel(ROOT,'data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz','output/reports/hierarchical_asset_itransformer_v43_clean_calendar_metadata.json')
    gaps.to_csv(ROOT / 'output/reports/v44_accounting_fixed_valuation_gaps.csv',index=False,encoding='utf-8-sig')
    complete_end = panel.index.max() - pd.offsets.MonthEnd(1)
    monthly = (1 + panel).resample('ME').prod(min_count=1) - 1
    counts = panel.notna().resample('ME').sum()
    expected = panel.index.to_series().resample('ME').size()
    monthly = monthly.where(counts.eq(expected,axis=0)).loc[:complete_end]
    records = []
    for asset in monthly:
        valid = monthly[asset].dropna()
        close = 100*(1+valid).cumprod()
        records.extend(dict(date=date,asset_code=asset,asset_name=asset,module=modules[asset],
                            return_1m=float(value),close=float(close.loc[date])) for date,value in valid.items())
    monthly_path = ROOT / 'data_processed/v44_frozen_returns_monthly.parquet'
    pd.DataFrame(records).to_parquet(monthly_path,index=False)
    panel.to_parquet(ROOT / 'data_processed/v44_frozen_returns_daily.parquet')
    long = panel.rename_axis(index='date',columns='asset_code').stack(future_stack=True).rename('return_1d').reset_index()
    long['module'] = long.asset_code.map(modules)
    long['asset_name'] = long.asset_code
    long['tradable'] = tradable.rename_axis(index='date',columns='asset_code').stack().to_numpy(dtype=bool)
    long.to_parquet(ROOT / 'data_processed/v44_frozen_returns_long.parquet',index=False)
    view_path = ROOT/'data_processed/v44_mechanical_views_daily.parquet'
    cards_path = ROOT/'data_processed/v44_state_cards_daily.parquet'
    for source,target in [('market_skill_views_mechanical_daily_v1.parquet',view_path),('market_state_cards_daily_v1.parquet',cards_path)]:
        data = pd.read_parquet(ROOT/'data_processed'/source)
        data['date'] = pd.to_datetime(data['date'])
        data = data.loc[data.date.isin(panel.index)].copy()
        data.to_parquet(target,index=False)
    paths = V1Paths(returns=monthly_path, views=view_path, market_state_cards=cards_path)
    base = load_config()
    for profile in ['low_vol','balanced','high_vol']:
        cfg = copy.deepcopy(base)
        cfg['product']['selected_profile'] = profile
        cfg['strategy_version'] = 'no_agent'
        cfg['rebalance']['end_date'] = str(complete_end.date())
        cfg['synthetic_assets']['rate_30y_curve_proxy']['enabled'] = False
        cfg.setdefault('signal_overlays',{}).setdefault('rule_output',{})['state_cards_path'] = str(cards_path.relative_to(ROOT))
        for key,value in cfg['outputs'].items():
            old = Path(value)
            cfg['outputs'][key] = str(old.parent / old.name.replace('v1_',f'v44_{profile}_'))
        weights = build_asset_weights(cfg,paths)
        if not np.allclose(weights.groupby('date').weight.sum(),1):
            raise RuntimeError('Rebuilt anchor weights do not sum to one')
        print(profile,'anchor_rows',len(weights),flush=True)
    manifest = {'accounting_version':ACCOUNTING_VERSION,'assets':len(panel.columns),
                'start':str(panel.index.min().date()),'end':str(panel.index.max().date()),
                'return_panel_hash':hashlib.sha256(pd.util.hash_pandas_object(panel,index=True).values.tobytes()).hexdigest(),
                'monthly_end':str(complete_end.date()),'source':'frozen v43 NPZ; bounded last-known valuation; no execution on unquoted days',
                'max_stale_sessions':5,
                'views':'mechanical only','ppo_status':'requires_new_training','forecast_status':'reuse causal saved predictions; formal gate remains authoritative'}
    (ROOT/'output/reports/v44_accounting_fixed_input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
