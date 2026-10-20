"""Boundary tests for the isolated v45 solver; no market training."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v45_solver import (solve, allocate_assets, bounded_projection,
    confidence_view_tilt, causal_anchor, SolverContractError, RISK_SLEEVES)


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.s = RISK_SLEEVES
        self.base = pd.Series(.2, index=self.s)
        self.mu = pd.Series([.04, .03, -.04, -.03, 0], index=self.s)
        self.cov = pd.DataFrame(np.eye(5)*.001, index=self.s, columns=self.s)
        self.hist = pd.DataFrame(np.random.default_rng(17).normal(0,.01,(100,5)), columns=self.s)
        self.bounds = {s:(0.,1.) for s in self.s}
        self.opts = dict(risk_aversion=3., base_deviation_penalty=0.,
            turnover_penalty=0., max_active_deviation=2., max_sleeve_turnover=.1)

    def run_solver(self, **kwargs):
        args = dict(base=self.base, mu=self.mu, covariance=self.cov,
            previous=self.base, bounds=self.bounds, history=self.hist, options=self.opts)
        args.update(kwargs)
        return solve(**args)

    def test_binding_total_turnover_and_reference(self):
        a = self.run_solver()
        b = self.run_solver(backend='cvxpy')
        self.assertEqual(a.diagnostics['backend'], 'skfolio')
        self.assertLessEqual(a.diagnostics['target_change_l1'], .100001)
        self.assertAlmostEqual(a.diagnostics['target_change_l1'], .1, places=6)
        self.assertLess(abs(a.diagnostics['objective']-b.diagnostics['objective']), 1e-8)

    def test_external_covariance(self):
        cov = self.cov.copy()
        cov.iloc[0,0] = .000001
        r = self.run_solver(mu=self.mu*0, covariance=cov,
            options={**self.opts,'max_sleeve_turnover':2.})
        self.assertGreater(r.weights.iloc[0], .99)

    def test_previous_not_projected(self):
        previous = pd.Series([.8,.05,.05,.05,.05],index=self.s)
        with self.assertRaises(SolverContractError) as e:
            self.run_solver(previous=previous, bounds={s:(0,.4) for s in self.s})
        self.assertEqual(e.exception.status, 'infeasible')
        self.assertGreater(e.exception.detail['minimum_required_l1'], .79)
        self.assertEqual(previous.iloc[0], .8)

    def test_anchor_jump_not_relaxed(self):
        previous = pd.Series([1,0,0,0,0],index=self.s,dtype=float)
        with self.assertRaises(SolverContractError) as e:
            self.run_solver(previous=previous,
                options={**self.opts,'max_active_deviation':.02})
        self.assertEqual(e.exception.status, 'infeasible')

    def test_causal_anchor_repairs_jump_without_raising_caps(self):
        p=pd.Series([.35,.365,.03,.06,.195],index=self.s)
        b=pd.Series([.287,.435,.03,.06,.188],index=self.s)
        a=causal_anchor(b,p,self.bounds,.022,.08)
        self.assertTrue(a.diagnostics['anchor_adjusted'])
        r=self.run_solver(base=a.weights,previous=p,
            options={**self.opts,'max_active_deviation':.022,'max_sleeve_turnover':.08})
        self.assertLessEqual(float((r.weights-p).abs().sum()),.080001)
        self.assertLessEqual(float((r.weights-a.weights).abs().sum()),.022001)
        np.testing.assert_allclose(b,[.287,.435,.03,.06,.188])

    def test_feasible_anchor_unchanged(self):
        a=causal_anchor(self.base,self.base,self.bounds,.022,.08)
        self.assertFalse(a.diagnostics['anchor_adjusted'])
        np.testing.assert_array_equal(a.weights,self.base)

    def test_new_guardrail_can_project_raw_anchor(self):
        raw=pd.Series([.45,.15,.15,.15,.1],index=self.s)
        limits={s:(0,.4 if s==self.s[0] else 1) for s in self.s}
        r=causal_anchor(raw,self.base,limits,.02,.5)
        self.assertTrue(r.diagnostics['anchor_adjusted'])
        self.assertLessEqual(r.weights.iloc[0],.400001)

    def test_new_guardrail_unreachable_still_blocks(self):
        prev=pd.Series([.8,.05,.05,.05,.05],index=self.s)
        limits={s:(0,.4) for s in self.s}
        with self.assertRaises(SolverContractError):
            causal_anchor(self.base,prev,limits,.02,.1)

    def test_infeasible_box(self):
        with self.assertRaises(SolverContractError):
            self.run_solver(bounds={s:(.3,1) for s in self.s})
        with self.assertRaises(SolverContractError):
            bounded_projection(np.ones(5),np.zeros(5),np.full(5,.1))

    def test_bad_labels(self):
        with self.assertRaises(SolverContractError):
            self.run_solver(covariance=self.cov.rename(columns={self.s[0]:'missing'}))
        with self.assertRaises(SolverContractError):
            self.run_solver(mu=self.mu.iloc[:-1])

    def test_nan_and_asymmetric_covariance(self):
        cov=self.cov.copy(); cov.iloc[0,1]=np.nan
        with self.assertRaises(SolverContractError): self.run_solver(covariance=cov)
        cov.iloc[0,1]=.0001
        with self.assertRaises(SolverContractError): self.run_solver(covariance=cov)

    def test_nonfinite_history_does_not_fallback(self):
        h=self.hist.copy(); h.iloc[0,0]=np.nan
        with self.assertRaises(SolverContractError) as e:
            self.run_solver(history=h)
        self.assertEqual(e.exception.status,'invalid_input')

    def test_zero_view_and_near_singular_pd(self):
        c=pd.DataFrame(np.eye(5)*1e-12,index=self.s,columns=self.s)
        r=self.run_solver(mu=self.mu*0,covariance=c)
        self.assertTrue(np.isfinite(r.weights).all())
        self.assertAlmostEqual(r.weights.sum(),1.,places=8)

    def test_covariance_reordered_labels(self):
        a=self.run_solver()
        b=self.run_solver(covariance=self.cov.iloc[::-1,::-1])
        np.testing.assert_allclose(a.weights,b.weights,atol=1e-7)

    def test_forced_primary_failure(self):
        with patch('src.v45_solver.MeanRisk.fit',side_effect=RuntimeError('forced')):
            r=self.run_solver()
        self.assertEqual(r.diagnostics['status'],'fallback_validated')
        self.assertEqual(r.diagnostics['backend'],'cvxpy')
        self.assertLessEqual(r.diagnostics['target_change_l1'], .100001)
        self.assertIn('forced',str(r.diagnostics['errors']))

    def test_no_unsafe_base_fallback(self):
        previous=pd.Series([.5,.2,.1,.1,.1],index=self.s)
        with patch('src.v45_solver.MeanRisk.fit',side_effect=RuntimeError('forced')):
            r=self.run_solver(previous=previous,
                options={**self.opts,'max_sleeve_turnover':.02})
        self.assertLessEqual(float((r.weights-previous).abs().sum()),.020001)

    def test_all_backends_failed_hold_only_if_feasible(self):
        with patch('src.v45_solver.MeanRisk.fit',side_effect=RuntimeError('forced')), \
                patch('src.v45_solver.cp.Problem.solve',side_effect=RuntimeError('forced')):
            r=self.run_solver()
        self.assertEqual(r.diagnostics['status'],'hold_validated')
        np.testing.assert_allclose(r.weights,self.base)

    def test_all_backends_failed_and_previous_illegal_blocks(self):
        p=pd.Series([.8,.05,.05,.05,.05],index=self.s)
        with patch('src.v45_solver.MeanRisk.fit',side_effect=RuntimeError('forced')), \
                patch('src.v45_solver.cp.Problem.solve',side_effect=RuntimeError('forced')):
            with self.assertRaises(SolverContractError) as e:
                self.run_solver(previous=p,bounds={s:(0,.4) for s in self.s},
                    options={**self.opts,'max_sleeve_turnover':2.})
        self.assertEqual(e.exception.status,'blocked')

    def test_penalized_reference_objective(self):
        o={**self.opts,'base_deviation_penalty':.3,'turnover_penalty':.02}
        a=self.run_solver(options=o)
        b=self.run_solver(options=o,backend='cvxpy')
        self.assertLess(abs(a.diagnostics['objective']-b.diagnostics['objective']),1e-8)

    def test_binding_active_and_box(self):
        r=self.run_solver(bounds={s:(.18,.22) for s in self.s},
            options={**self.opts,'max_active_deviation':.03})
        self.assertLessEqual(float((r.weights-self.base).abs().sum()),.030001)
        self.assertTrue(r.weights.between(.18-1e-6,.22+1e-6).all())

    def test_asset_cap_and_budget(self):
        b=pd.Series([.7,.2,.1],index=['a','b','c'])
        r=allocate_assets(b,pd.Series([3.,0.,0.],index=b.index),b*0,.2,1,.75,.02)
        self.assertAlmostEqual(r.sum(),.2,places=8)
        self.assertLessEqual(r.max(),.150001)

    def test_asset_cap_intersection_infeasible(self):
        b=pd.Series([.9,.1],index=['a','b'])
        with self.assertRaises(SolverContractError):
            allocate_assets(b,b*0,b*0,.2,.08,.75,.004)

    def test_zero_anchor_asset_not_added(self):
        b=pd.Series([.5,.5,0.],index=['a','b','c'])
        r=allocate_assets(b,pd.Series([0.,0.,100.],index=b.index),b*0,.2,1,.75,.01)
        self.assertEqual(r['c'],0)

    def test_low_confidence_not_amplified(self):
        p=pd.DataFrame([dict(source='equity',target='rates',confidence=.01)])
        low=confidence_view_tilt(p,.004)
        p['confidence']=.99
        high=confidence_view_tilt(p,.004)
        self.assertAlmostEqual(high.equity/low.equity,99)

    def test_duplicate_and_conflict_views(self):
        p=pd.DataFrame([dict(source='equity',target='rates',confidence=.5)])
        np.testing.assert_allclose(confidence_view_tilt(p,.004),
            confidence_view_tilt(pd.concat([p,p]),.004))
        p.loc[1]=['rates','equity',.5]
        self.assertTrue((confidence_view_tilt(p,.004)==0).all())

    def test_invalid_confidence_and_negative_options(self):
        p=pd.DataFrame([dict(source='equity',target='rates',confidence=1.1)])
        with self.assertRaises(SolverContractError): confidence_view_tilt(p,.004)
        with self.assertRaises(SolverContractError):
            self.run_solver(options={**self.opts,'risk_aversion':-1})


if __name__=='__main__':
    unittest.main(verbosity=2)
