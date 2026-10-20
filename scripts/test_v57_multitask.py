import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.v57_multitask import future_downside, DistributionHead, multitask_loss, execution_observation, require_execution_certificate


class RebuildTests(unittest.TestCase):
    def test_tail_missing_and_forward_path(self):
        r=np.array([[0.],[.1],[-.2],[.1],[0.]])
        y,m=future_downside(r,np.ones_like(r,bool),[2])
        self.assertAlmostEqual(float(y[0,0,0]),.2,places=6)
        self.assertFalse(m[-2:].any())

    def test_missing_is_not_zero_return(self):
        r=np.zeros((8,2));m=np.ones_like(r,bool);m[2,0]=False
        _,good=future_downside(r,m,[3])
        self.assertFalse(good[0,0,0]);self.assertTrue(good[0,0,1])

    def test_labels_exclude_signal_day(self):
        r=np.zeros((10,2));m=np.ones_like(r,bool)
        a,_=future_downside(r,m,[3]);r[0]=-.9
        b,_=future_downside(r,m,[3]);np.testing.assert_equal(a[0],b[0])

    def test_order_and_gradients(self):
        head=DistributionHead();q,d,mean=head(torch.randn(4,5,40))
        self.assertEqual(tuple(q.shape),(4,3,5,3))
        self.assertTrue((q[...,0]<=q[...,1]).all() and (q[...,1]<=q[...,2]).all())
        y=torch.zeros(4,3,5);mask=torch.ones_like(y,dtype=torch.bool)
        loss=multitask_loss(q,d,mean,y,mask,y,mask,torch.zeros(5,dtype=torch.long))
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_observation_excludes_future_reward(self):
        obs=execution_observation([.01,.02],[0,0],[.03,.04],[.02,.03],[.5,.5],[.6,.4],4,5)
        self.assertAlmostEqual(float(obs[-1]),.2*.0005,places=8)
        with self.assertRaises(ValueError):execution_observation([0],[1],[0],[0],[1],[1],0,5)

    def test_uncertified_execution_fails_closed(self):
        with self.assertRaises(ValueError):require_execution_certificate(None,"x",["a"],ROOT)
        with self.assertRaises(ValueError):require_execution_certificate({"dataset_sha256":"x"},"x",["a"],ROOT)


if __name__=="__main__":unittest.main()
