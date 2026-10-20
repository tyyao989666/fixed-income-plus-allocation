"""Behavioral regression tests for repaired portfolio accounting and risk inputs."""
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.portfolio_accounting import advance_holdings, calendar_decision_targets, frozen_return_panel, accounting_return_panel
from src.adaptive_daily_ppo import perf_stats
from src.ep_skfolio_optimizer import optimize_sleeve_weights, RISK_SLEEVES
from src.hierarchical_asset_portfolio_v7 import AdaptiveFractionEnvironment, controller_signature, load_controller_models, train_controller_seed
from src.hierarchical_asset_itransformer import load_config as load_model_config

spec = importlib.util.spec_from_file_location('baselines', ROOT / 'scripts/171_run_v43_control_variable_baselines.py')
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)


class AccountingTests(unittest.TestCase):
    def test_drift_and_no_hidden_rebalance(self):
        first = advance_holdings([.5,.5],[.5,.5],0,[.1,0],.0005)
        self.assertAlmostEqual(first['end_weights'][0], .55/1.05)
        second = advance_holdings(first['end_weights'],[.5,.5],0,[0,.1],.0005)
        self.assertEqual(second['turnover'],0)
        self.assertAlmostEqual(second['gross_return'], (.5/1.05)*.1)

    def test_trade_cost_and_drift(self):
        step = advance_holdings([.5,.5],[1,0],.5,[.1,0],.0005)
        self.assertAlmostEqual(step['turnover'],.5)
        self.assertAlmostEqual(step['cost'],.00025)
        self.assertAlmostEqual(step['return'],.07475)
        self.assertAlmostEqual(step['end_weights'].sum(),1)

    def test_missing_held_return_fails(self):
        with self.assertRaisesRegex(ValueError,'Missing execution'):
            advance_holdings([.5,.5],[.5,.5],0,[np.nan,0],.0005)
        self.assertEqual(advance_holdings([0,1],[0,1],0,[np.nan,.1],.0005)['return'],.1)

    def test_nominal_weekend_month_end_is_friday_signal(self):
        calendar = pd.DatetimeIndex(['2022-04-29','2022-05-05','2022-05-06'])
        source = pd.DataFrame({'a':[1.]},index=pd.DatetimeIndex(['2022-04-30'],name='date'))
        mapped = calendar_decision_targets(source,calendar)
        self.assertEqual(mapped.iloc[0].a,1)

    def test_unquoted_asset_cannot_be_bought(self):
        step = advance_holdings([.5,.5],[1,0],1,[0,.1],.0005,[False,True])
        np.testing.assert_allclose(step['weights'],[.5,.5])
        self.assertTrue(step['unfilled'])
        self.assertEqual(step['turnover'],0)

    def test_bounded_valuation_does_not_backfill_before_listing(self):
        dates = pd.date_range('2022-01-01',periods=8)
        values = pd.DataFrame({'a':[np.nan,.1,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]},index=dates)
        with patch('src.portfolio_accounting.frozen_return_panel',return_value=(values,pd.Series({'a':'equity'}))):
            marked,_,tradable,gaps = accounting_return_panel(None,None,None,max_stale_sessions=5)
        self.assertTrue(pd.isna(marked.iloc[0,0]))
        self.assertEqual(marked.iloc[6,0],0)
        self.assertTrue(pd.isna(marked.iloc[7,0]))
        self.assertFalse(tradable.iloc[6,0])

    def test_standard_sharpe_and_initial_drawdown(self):
        r = pd.Series([-.03,.01,.02,-.01])
        result = perf_stats(r)
        self.assertAlmostEqual(result['sharpe'],r.mean()/r.std(ddof=1)*np.sqrt(252))
        self.assertLessEqual(result['max_drawdown'],-.03)

    def test_formal_signal_t1_and_passive_drift(self):
        dates = pd.DatetimeIndex(['2022-01-04','2022-01-05','2022-01-06'],name='date')
        returns = pd.DataFrame({'a':[.1,0,0],'b':[0,.1,.1]},index=dates)
        source = pd.DataFrame({'date':pd.to_datetime(['2021-12-31','2021-12-31','2022-01-05','2022-01-05']),
                               'asset_code':['a','b','a','b'],'module':['equity']*4,'weight':[.5,.5,1,0]})
        with patch.object(baselines.pd,'read_parquet',return_value=source), patch.object(baselines,'formal_rate_split',return_value=pd.Series(dtype=float)):
            daily, _ = baselines.simulate_formal_anchor(returns,'low_vol')
        self.assertAlmostEqual(daily.iloc[1]['gross_return'],.1*.5/1.05)
        self.assertEqual(daily.iloc[1]['turnover'],0)
        self.assertEqual(daily.iloc[2]['gross_return'],0)
        self.assertGreater(daily.iloc[2]['turnover'],0)

    def test_risk_parity_passive_drift(self):
        dates = pd.DatetimeIndex(['2022-01-04','2022-01-05','2022-01-06'],name='date')
        returns = pd.DataFrame({'a':[.1,0,0],'b':[0,.1,.1]},index=dates)
        with patch.object(baselines,'risk_parity_target',return_value=pd.Series([.5,.5],index=['a','b'])):
            daily, _ = baselines.simulate_monthly_risk_parity(returns,returns,pd.Series({'a':'equity','b':'equity'}),{})
        self.assertAlmostEqual(daily.iloc[1]['gross_return'],.1*.5/1.05)
        self.assertEqual(daily.iloc[1]['turnover'],0)

    def test_skfolio_uses_external_covariance_and_matches_fallback(self):
        names = RISK_SLEEVES
        base = pd.Series(.2,index=names)
        mu = pd.Series(0.,index=names)
        history = pd.DataFrame(np.random.default_rng(17).normal(0,.01,(100,5)),columns=names)
        cov = pd.DataFrame(np.diag([.001,.1,.1,.1,.1]),index=names,columns=names)
        bounds = {key:(0,1) for key in names}
        options = {'base_deviation_penalty':0,'turnover_penalty':0,'max_active_deviation':2,'max_sleeve_turnover':2,'tolerance':1e-12}
        cvx = optimize_sleeve_weights(base,mu,cov,history,base,bounds,{'ep_skfolio':options})
        self.assertEqual(cvx.diagnostics['optimizer_backend'],'skfolio_meanrisk_clarabel',cvx.diagnostics)
        self.assertGreater(cvx.weights.iloc[0],.9)
        changed = cov.iloc[::-1,::-1].copy()
        changed.index,changed.columns = names,names
        other = optimize_sleeve_weights(base,mu,changed,history,base,bounds,{'ep_skfolio':options})
        self.assertGreater(other.weights.iloc[-1],.9)
        fallback = optimize_sleeve_weights(base,mu,cov,history,base,bounds,{'ep_skfolio':{**options,'backend':'scipy'}})
        np.testing.assert_allclose(cvx.weights,fallback.weights,atol=1e-3)

    def test_skfolio_regularized_objective_matches_fallback(self):
        names = RISK_SLEEVES
        base = pd.Series(.2,index=names)
        mu = pd.Series([.04,.01,.03,0,-.01],index=names)
        history = pd.DataFrame(np.random.default_rng(17).normal(0,.01,(100,5)),columns=names)
        cov = pd.DataFrame(np.diag([.001,.1,.03,.04,.02]),index=names,columns=names)
        bounds = {key:(0,1) for key in names}
        options = {'base_deviation_penalty':.3,'turnover_penalty':.02,'max_active_deviation':1,'max_sleeve_turnover':1,'tolerance':1e-12}
        cvx = optimize_sleeve_weights(base,mu,cov,history,base,bounds,{'ep_skfolio':options})
        fallback = optimize_sleeve_weights(base,mu,cov,history,base,bounds,{'ep_skfolio':{**options,'backend':'scipy'}})
        np.testing.assert_allclose(cvx.weights,fallback.weights,atol=1e-3)

    def environment(self):
        dates = pd.DatetimeIndex(['2022-12-29','2022-12-30','2023-01-03','2023-01-04'],name='date')
        returns = pd.DataFrame({'a':[0,.1,0,0],'b':[0,0,.1,0]},index=dates)
        formal = pd.DataFrame({'a':.5,'b':.5},index=dates)
        target = pd.DataFrame([{'date':date,'asset_code':asset,'module':'rates' if asset=='a' else 'equity','weight':.5} for date in dates for asset in ['a','b']])
        state = pd.DataFrame({'date':dates,'state_bond_score':.5})
        diagnostics = pd.DataFrame({'date':dates,'forecast_confidence':0.})
        cfg = {'controller':{'action_fractions':[0,.25,.5,1],'active_weight_penalty':0},'portfolio':{'transaction_cost_bps':5},'data':{'horizons_days':[5,20,60]},'outputs':{'dataset_npz':'unused','metadata':'unused'}}
        with patch('src.hierarchical_asset_portfolio_v7.accounting_return_panel',return_value=(returns,None,returns.notna(),pd.DataFrame())):
            return AdaptiveFractionEnvironment('low_vol',cfg,target,diagnostics,state,formal)

    def test_ppo_formal_benchmark_drifts_without_hidden_trading(self):
        env = self.environment()
        reward,_,weights,context,first = env.step(0,0,env.initial_weights(0),env.initial_context())
        reward,_,weights,context,second = env.step(1,0,weights,context)
        self.assertAlmostEqual(second['formal_return'],.1*.5/1.05)
        self.assertEqual(second['formal_turnover'],0)
        self.assertAlmostEqual(second['formal_return'],second['strategy_return'])

    def test_checkpoint_signature_ignores_future_data(self):
        env = self.environment()
        original = controller_signature(env,2023)
        env.returns.loc[env.returns.index >= '2023-01-01','a'] = .2
        self.assertEqual(original,controller_signature(env,2023))
        env.returns.iloc[0,0] = .1
        self.assertNotEqual(original,controller_signature(env,2023))

    def test_old_checkpoints_rejected(self):
        import torch
        env = self.environment()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'policy.pt'
            torch.save([{'state_dict':{}}],path)
            with self.assertRaisesRegex(RuntimeError,'Stale PPO'):
                load_controller_models(env,2023,path)

    def test_teacher_and_ppo_training_smoke(self):
        import torch
        torch.set_num_threads(1)
        dates = pd.bdate_range('2018-01-01',periods=180,name='date')
        returns = pd.DataFrame(np.random.default_rng(17).normal(0,.001,(180,2)),index=dates,columns=['a','b'])
        formal = pd.DataFrame({'a':.5,'b':.5},index=dates)
        target = pd.DataFrame([{'date':date,'asset_code':asset,'module':'rates' if asset=='a' else 'equity',
                               'weight':(.6 if asset=='a' else .4)} for date in dates for asset in ['a','b']])
        diagnostics = pd.DataFrame({'date':dates,'forecast_confidence':.5})
        state = pd.DataFrame({'date':dates,'state_bond_score':.5})
        cfg = load_model_config(ROOT/'config/hierarchical_asset_itransformer_v44_accounting_fixed.yaml')
        cfg['controller']['max_epochs'] = 1
        cfg['controller']['teacher_guidance']['pretrain_epochs'] = 1
        with patch('src.hierarchical_asset_portfolio_v7.accounting_return_panel',return_value=(returns,None,returns.notna(),pd.DataFrame())):
            env = AdaptiveFractionEnvironment('low_vol',cfg,target,diagnostics,state,formal)
        checkpoint,logs = train_controller_seed(env,env.eligible(pd.Timestamp('2019-01-01')),2019,17)
        self.assertEqual(checkpoint['accounting_signature'],controller_signature(env,2019))
        self.assertTrue(all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values()))
        self.assertTrue(any(row['epoch']=='deployment_gate' for row in logs))

    def test_snapshot_has_38_identical_named_assets(self):
        returns, modules = frozen_return_panel(ROOT,'data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz','output/reports/hierarchical_asset_itransformer_v43_clean_calendar_metadata.json')
        self.assertEqual(len(returns.columns),38)
        self.assertIn('rate_30y_curve_proxy',returns)
        self.assertNotIn('rate_30y',returns)
        self.assertFalse((returns.index.dayofweek>=5).any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
