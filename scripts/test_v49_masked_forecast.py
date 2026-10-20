"""Mask, chronology and true-gradient tests for the new research encoder."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src import v45_forecast as old
from src.v49_masked_forecast import MaskSafeForecastNetwork, safe_batch_at, validate_dataset, training_implementation


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.a = old.synthetic_arrays(rows=400)
        cls.cfg = copy.deepcopy(old.DEFAULTS)

    def test_strict_schema_and_endpoints(self):
        validate_dataset(self.a)
        bad = copy.deepcopy(self.a)
        bad["label_end"][200, 0] = bad["dates"][206]
        with self.assertRaisesRegex(ValueError, "endpoint"):
            validate_dataset(bad)

    def test_invalid_tail_labels_rejected(self):
        bad = copy.deepcopy(self.a)
        bad["asset_target_mask"][-1, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "Unmatured"):
            validate_dataset(bad)

    def test_nonbinary_and_nonfinite_observed_rejected(self):
        bad = copy.deepcopy(self.a)
        bad["local_masks"] = bad["local_masks"].astype(float)
        bad["local_masks"][200, 0, 0] = .5
        with self.assertRaisesRegex(ValueError, "Nonbinary"):
            validate_dataset(bad)
        bad = copy.deepcopy(self.a)
        bad["local_masks"][200, 0, 0] = True
        bad["local_values"][200, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid observed local"):
            validate_dataset(bad)

    def test_window_boundary_rejected(self):
        for indices in [np.array([118]), np.array([400]), np.array([200.5])]:
            with self.assertRaises(ValueError):
                safe_batch_at(self.a, indices, self.cfg, "cpu")

    def test_future_suffix_does_not_affect_observation(self):
        after = copy.deepcopy(self.a)
        after["local_values"][201:] = 1e6
        after["macro_values"][201:] = 1000.
        for x, y in zip(safe_batch_at(self.a, np.array([200]), self.cfg, "cpu"),
                        safe_batch_at(after, np.array([200]), self.cfg, "cpu")):
            torch.testing.assert_close(x, y)

    def test_masked_payload_cannot_affect_forecast(self):
        net = MaskSafeForecastNetwork(self.a, self.cfg).eval()
        x, m, macro = safe_batch_at(self.a, np.array([200]), self.cfg, "cpu")
        dirty = x.clone()
        dirty[~m.bool()] = float("nan")
        with torch.no_grad():
            a, b = net((x, m, macro)), net((dirty, m, macro))
        for task in a:
            torch.testing.assert_close(a[task], b[task])

    def test_unobserved_identity_cannot_pollute_observed_assets(self):
        net = MaskSafeForecastNetwork(self.a, self.cfg).eval()
        x, m, macro = safe_batch_at(self.a, np.array([200]), self.cfg, "cpu")
        x[:, :, 0], m[:, :, 0] = 0., 0.
        with torch.no_grad():
            before = net((x, m, macro))
            net.core.asset_embedding.weight[0] += torch.linspace(-50, 50, self.cfg["d_model"])
            after = net((x, m, macro))
        torch.testing.assert_close(before["asset"][:, :, 1:], after["asset"][:, :, 1:])
        torch.testing.assert_close(before["sleeve"], after["sleeve"])

    def test_all_missing_and_empty_sleeve_are_finite(self):
        net = MaskSafeForecastNetwork(self.a, self.cfg).eval()
        x, m, macro = safe_batch_at(self.a, np.array([200]), self.cfg, "cpu")
        m.zero_()
        with torch.no_grad():
            encoded = net.encode((x, m, macro))
            pred = net((x, m, macro))
        self.assertTrue(torch.count_nonzero(encoded[0]) == 0)
        self.assertTrue(torch.count_nonzero(encoded[3]) == 0)
        self.assertTrue(all(torch.isfinite(v).all() for v in pred.values()))

    def test_all_observed_parity_with_legacy(self):
        net = MaskSafeForecastNetwork(self.a, self.cfg).eval()
        legacy = old.ForecastNetwork(self.a, self.cfg).eval()
        legacy.load_state_dict(net.state_dict())
        x, m, macro = safe_batch_at(self.a, np.array([200]), self.cfg, "cpu")
        m.fill_(1.)
        with torch.no_grad():
            before, after = legacy((x, m, macro)), net((x, m, macro))
        for task in before:
            torch.testing.assert_close(before[task], after[task], atol=2e-6, rtol=2e-5)

    def test_actual_ssl_and_supervised_gradients(self):
        net = MaskSafeForecastNetwork(self.a, self.cfg).train()
        batch = safe_batch_at(self.a, np.array([200, 205]), self.cfg, "cpu")
        before = net.core.encoder.layers[0].linear1.weight.detach().clone()
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
        loss = net.reconstruction_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(net.core.encoder.layers[0].linear1.weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, net.core.encoder.layers[0].linear1.weight))
        optimizer.zero_grad(set_to_none=True)
        pred = net(batch)
        targets, masks = old.targets_at(self.a, np.array([200, 205]))
        loss = old.group_loss(pred, {k:torch.tensor(v) for k,v in targets.items()},
                              {k:torch.tensor(v) for k,v in masks.items()},
                              torch.tensor(self.a["module_index"]), [.15, .5, .35])
        loss.backward()
        self.assertTrue(torch.isfinite(net.asset_output.weight.grad).all())

    def test_trainer_uses_new_implementation_without_recursion_or_permanent_patch(self):
        prior = old.ForecastNetwork
        with training_implementation():
            net = old.ForecastNetwork(self.a, self.cfg)
            self.assertIsInstance(net, MaskSafeForecastNetwork)
            batch = old.batch_at(self.a, np.array([200]), self.cfg, "cpu")
            self.assertEqual(batch[0].shape[1], 120)
            with self.assertRaisesRegex(ValueError, "spawned legacy workers"):
                list(old.seed_worker_results({}, [17], 2))
        self.assertIs(old.ForecastNetwork, prior)


if __name__ == "__main__":
    unittest.main()
