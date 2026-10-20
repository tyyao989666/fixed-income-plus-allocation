import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import v45_forecast as engine
from src.v49_masked_forecast import MaskSafeCore
from src.v60_supervised_adapter import SupervisedAdapter, capture_prefix, fit_prior, supervised_loss


class AdapterTests(unittest.TestCase):
    def fixture(self):
        cfg = engine.configuration(Path(__file__).resolve().parents[1] / "config/v45_research.yaml")
        core = MaskSafeCore(120, 3, 2, np.arange(5), 3, cfg).eval()
        class Network(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
            def encode(self, batch):
                return self.core.encode_context(*batch)
        x = torch.randn(3, 120, 5, 3)
        mask = torch.ones_like(x)
        mask[:, :, 4] = 0
        batch = x, mask, torch.randn(3, 120, 2)
        prefix, expected = capture_prefix(Network(core), batch)
        priors = {t: np.zeros((3, 5), np.float32) for t in ["asset", "sleeve"]}
        scales = {t: np.ones((3, 5), np.float32) for t in priors}
        return core, prefix, expected, priors, scales

    def test_cached_replay_matches_original(self):
        core, prefix, expected, prior, scale = self.fixture()
        model = SupervisedAdapter(core, prior, prior, scale, "frozen_control").eval()
        with torch.no_grad():
            actual = model.representations(prefix)
        for task in expected:
            torch.testing.assert_close(actual[task], expected[task], atol=1e-5, rtol=1e-5)

    def test_zero_initial_dynamic_increment(self):
        core, prefix, _, prior, scale = self.fixture()
        model = SupervisedAdapter(core, prior, prior, scale, "last_block_adapter")
        model.train()
        self.assertFalse(model.core.encoder.layers[0].training)
        self.assertTrue(model.core.encoder.layers[-1].training)
        mean, delta = model(prefix)
        for t in prior:
            self.assertTrue((delta[t] == 0).all())
            self.assertTrue((model.decode(mean)[t] == 0).all())

    def test_only_last_block_and_heads_receive_updates(self):
        core, prefix, _, prior, scale = self.fixture()
        model = SupervisedAdapter(core, prior, prior, scale, "last_block_adapter")
        initial = {k:v.clone() for k,v in model.named_parameters()}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        y = {t: torch.randn(3, 3, 5) for t in prior}
        m = {t: torch.ones_like(v, dtype=torch.bool) for t,v in y.items()}
        g = dict(asset=torch.arange(5), sleeve=torch.zeros(5, dtype=torch.long))
        for step in range(3):
            opt.zero_grad()
            pred, d = model(prefix)
            loss = supervised_loss(pred, d, y, m, g, dict(rank_weight=.05, residual_penalty=.01))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            opt.step()
        changed = [k for k,p in model.named_parameters() if not torch.equal(p, initial[k])]
        self.assertTrue(any(k.startswith("core.encoder.layers.1.") for k in changed))
        self.assertTrue(all(k.startswith(("heads.", "core.encoder.layers.1.")) for k in changed))

    def test_prior_T_only_and_missing(self):
        y = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
        m = np.ones_like(y, bool)
        m[:, :, 1] = False
        prior, c, s, available, counts = fit_prior(y[:3], m[:3], [0, 0])
        np.testing.assert_array_equal(prior[:, 1], c[:, 1])
        self.assertTrue((counts[:, 1] == 0).all())
        y[3:] = 1e20
        prior2 = fit_prior(y[:3], m[:3], [0, 0])[0]
        np.testing.assert_array_equal(prior, prior2)


if __name__ == "__main__":
    unittest.main()
