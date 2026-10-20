"""Validate saved repaired output and write an explicit mechanical-preview report."""
from pathlib import Path
import hashlib
import json
import sys
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.portfolio_accounting import accounting_return_panel

PREFIX = 'hierarchical_asset_itransformer_v44_accounting_fixed'


def main():
    reports = ROOT/'output/reports'
    panel,_,tradable,gaps = accounting_return_panel(ROOT,'data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz','output/reports/hierarchical_asset_itransformer_v43_clean_calendar_metadata.json')
    checks = []
    for profile in ['low_vol','balanced','high_vol']:
        daily = pd.read_csv(reports/f'{PREFIX}_{profile}_daily.csv',parse_dates=['date','signal_date'])
        def add(name,passed,value=''):
            checks.append({'profile':profile,'check':name,'passed':bool(passed),'value':value})
        a0 = pd.read_csv(reports/f'v44_accounting_fixed_{profile}_A0_hierarchical_risk_parity_daily.csv',parse_dates=['date'])
        a1 = pd.read_csv(reports/f'v44_accounting_fixed_{profile}_A1_market_view_ep_skfolio_daily.csv',parse_dates=['date'])
        add('all_layers_same_execution_calendar',daily.date.equals(a0.date) and daily.date.equals(a1.date),len(daily))
        positions = panel.index.get_indexer(daily.signal_date)
        add('exact_next_session',bool((positions>=0).all()) and np.array_equal(panel.index[positions+1].to_numpy(),daily.date.to_numpy()))
        add('finite_returns',np.isfinite(daily.strategy_return).all())
        add('5bp_l1_cost',np.allclose(daily.transaction_cost,daily.turnover*.0005))
        add('formal_5bp_l1_cost',np.allclose(daily.formal_cost,daily.formal_turnover*.0005))
        weights = pd.read_parquet(ROOT/f'data_processed/{PREFIX}_{profile}_weights_daily.parquet').pivot(index='date',columns='asset_code',values='weight').sort_index()
        targets = pd.read_parquet(ROOT/f'data_processed/{PREFIX}_{profile}_targets_daily.parquet')
        add('weights_sum_one',np.allclose(weights.sum(axis=1),1))
        add('targets_sum_one',np.allclose(targets.groupby('date').weight.sum(),1))
        add('nonnegative_weights',weights.ge(-1e-10).all().all())
        add('no_cash','cash' not in weights)
        returns = panel.reindex(index=weights.index,columns=weights.columns)
        prior_end = weights.shift(1)*(1+returns.shift(1))
        prior_end = prior_end.div(prior_end.sum(axis=1),axis=0)
        unavailable = ~tradable.reindex(index=weights.index,columns=weights.columns)
        delta = (weights-prior_end).where(unavailable).abs()
        maximum = float(delta.max().max()) if delta.notna().any().any() else 0
        add('unquoted_assets_not_traded',maximum<1e-8,maximum)
        add('no_held_overage_gap',not ((returns.isna()) & weights.gt(1e-10)).any().any())
        metrics = pd.read_csv(reports/f'{PREFIX}_{profile}_metrics.csv')
        row = metrics.loc[metrics.split.eq('oos_2024_onward')].iloc[0]
        sample = daily.loc[daily.date.ge('2024-01-01'),'strategy_return']
        sharp = sample.mean()/sample.std(ddof=1)*np.sqrt(252)
        add('standard_sharpe_matches_output',np.isclose(row.strategy_sharpe,sharp),sharp)
    frame = pd.DataFrame(checks)
    frame.to_csv(reports/'v44_accounting_fixed_validation.csv',index=False,encoding='utf-8-sig')
    versions = {}
    for name in ['numpy','pandas','pyarrow','scipy','scikit-learn','skfolio','cvxpy','clarabel','torch','PyYAML']:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = 'not_installed'
    (reports/'v44_accounting_fixed_environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    gate = json.loads((reports/f'{PREFIX}_portfolio_gate.json').read_text(encoding='utf-8'))
    summary = pd.read_csv(reports/f'{PREFIX}_three_profile_summary.csv')
    table = summary.loc[summary.split.eq('oos_2024_onward'),['profile','strategy_annual_return','strategy_annual_vol','strategy_sharpe','strategy_max_drawdown']].copy()
    for col in ['strategy_annual_return','strategy_annual_vol','strategy_max_drawdown']:
        table[col] = table[col].map(lambda x:f'{x:.2%}')
    ppo_count = sum(int(pd.read_csv(reports/f'{PREFIX}_{profile}_daily.csv').policy_source.eq('ppo').sum()) for profile in ['low_vol','balanced','high_vol'])
    checkpoints = list((ROOT/f'data_processed/{PREFIX}_controller_models').rglob('policy_*.pt'))
    mode = '无新检查点的机械预览' if not checkpoints else '修正版检查点训练/回放（是否实际放行须另看PPO执行日）'
    lines = ['# v44修复后输出检查','',f'检查通过：{int(frame.passed.sum())}/{len(frame)}。',
             f'本次模式：{mode}；PPO实际执行日总数：{ppo_count}。',
             f'正式门：{gate["enabled_as_formal"]}。','',
             '## 2024年以来保存结果','',table.to_markdown(index=False),'',
             '## 解释','',
             '这些数值用于确认修正版预测/目标/执行链路；旧PPO已隔离，完整PPO表现需使用--train-ppo重新训练后评价。',
             '缺口以最多5个交易观察日的延用估值记录，不可交易；原因尚未逐一验证。信用/REITs代理、宏观历史版本和未触碰测试仍是后续数据审计事项。',
             'A0与A1比较的是整套风险预算/资产选择/规则优化方案；严格单组件消融仍需后续独立实验，不能把全部差值归给单一Market View或EP模块。','',
             '环境版本见v44_accounting_fixed_environment.json，详细检查见v44_accounting_fixed_validation.csv。']
    (reports/'v44_accounting_fixed_validation.md').write_text('\n'.join(lines),encoding='utf-8')
    print(frame.to_string(index=False))
    print(table.to_string(index=False))
    if not frame.passed.all():
        raise SystemExit('Repaired output validation failed')


if __name__=='__main__':
    main()
