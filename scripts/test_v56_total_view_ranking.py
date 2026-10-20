"""Tests for the isolated ranking objective, no training or future labels."""
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v56_total_view_ranking import full_view_pair, total_view_ranking_implementation
from src import v45_forecast as engine


class RankingTests(unittest.TestCase):
    def fixture(self):
        p = torch.zeros(1, 3, 2, requires_grad=True)
        y = torch.tensor([[[-1., 1.], [-1., 1.], [-1., 1.]]])
        raw = np.array([[[3., 1.], [3., 1.], [3., 1.]]], dtype=np.float32)
        mask = torch.ones_like(y, dtype=torch.bool)
        return {"asset": p, "sleeve": p}, {"asset": y, "sleeve": y}, {"asset": raw, "sleeve": raw}, {"asset": np.ones((3, 2)), "sleeve": np.ones((3, 2))}, {"asset": mask, "sleeve": mask}

    def test_total_baseline_recovered(self):
        p, y, raw, scale, mask = self.fixture()
        total, target = full_view_pair(p, y, raw, scale, mask)
        self.assertTrue(torch.equal(total["asset"], torch.tensor([[[4., 0.]]*3])))
        self.assertTrue(torch.equal(target["asset"], torch.tensor(raw["asset"])))

    def test_gradient_direction_avoids_residual_rank_reversal(self):
        p, y, raw, scale, mask = self.fixture()
        total, target = full_view_pair(p, y, raw, scale, mask)
        old = torch.nn.functional.softplus(-(y["asset"][:,:,0]-y["asset"][:,:,1]).sign()*(p["asset"][:,:,0]-p["asset"][:,:,1])).mean()
        new = torch.nn.functional.softplus(-(target["asset"][:,:,0]-target["asset"][:,:,1]).sign()*(total["asset"][:,:,0]-total["asset"][:,:,1])).mean()
        old_grad = torch.autograd.grad(old, p["asset"], retain_graph=True)[0][0,0,0]
        new_grad = torch.autograd.grad(new, p["asset"])[0][0,0,0]
        self.assertGreater(old_grad.item(), 0)
        self.assertLess(new_grad.item(), 0)

    def test_masked_labels_not_exposed(self):
        p, y, raw, scale, mask = self.fixture()
        mask["asset"][:,:,1] = False
        raw["asset"][:,:,1] = np.nan
        y["asset"][:,:,1] = float("nan")
        total, target = full_view_pair(p, y, raw, scale, mask)
        self.assertTrue(torch.isfinite(total["asset"]).all())
        self.assertTrue((target["asset"][:,:,1] == 0).all())

    def test_context_restores_on_exception(self):
        original = engine.train_network
        try:
            with total_view_ranking_implementation(engine):
                self.assertIsNot(engine.train_network, original)
                raise RuntimeError("deliberate fixture")
        except RuntimeError:
            pass
        self.assertIs(engine.train_network, original)

    def test_scoped_wrapper_uses_only_captured_train_rows(self):
        p, y, raw, scale, mask = self.fixture()
        observed = {}
        fake = SimpleNamespace()
        fake.batch_at = lambda a, ix, cfg, device: ix

        def base_loss(pred, target, masks, modules, weights, pairwise=False):
            observed["total"] = pred["asset"].detach().clone()
            return pred["asset"].sum()

        def base_train(a, split, cfg, stage, scale):
            fake.batch_at(a, np.array([0]), cfg, "cpu")
            return fake.group_loss(p, y, mask, None, None, True)

        fake.group_loss, fake.train_network = base_loss, base_train
        with total_view_ranking_implementation(fake):
            fake.train_network({k+"_target": v for k,v in raw.items()},
                               {"train": [0]}, {"seed_workers": 1}, "F3", scale)
        self.assertTrue(torch.equal(observed["total"], torch.tensor([[[4.,0.]]*3])))
        self.assertIs(fake.group_loss, base_loss)
        self.assertIs(fake.train_network, base_train)

    def test_scoped_wrapper_rejects_nontrain_dates(self):
        p, y, raw, scale, mask = self.fixture()
        fake = SimpleNamespace(batch_at=lambda a, ix, cfg, device: ix,
                               group_loss=lambda *args: None)

        def base_train(a, split, cfg, stage, scale):
            fake.batch_at(a, np.array([1]), cfg, "cpu")
            return fake.group_loss(p, y, mask, None, None, True)

        fake.train_network = base_train
        prior_batch, prior_loss = fake.batch_at, fake.group_loss
        with total_view_ranking_implementation(fake):
            with self.assertRaisesRegex(ValueError, "outside captured training"):
                fake.train_network({k+"_target": v for k,v in raw.items()},
                                   {"train": [0]}, {"seed_workers": 1}, "F3", scale)
        self.assertIs(fake.batch_at, prior_batch)
        self.assertIs(fake.group_loss, prior_loss)


if __name__ == "__main__":
    unittest.main()
