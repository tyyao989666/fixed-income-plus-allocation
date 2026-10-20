"""Synthetic PIT/normalization/action-value contracts; no real model training."""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_v45_portfolio import fixture
from src.v48_execution_learning import (ScaledExecutionEnvironment, soft_targets,
    standardizer, training_normalizer, teacher_data, fit_seed)


def environment():
    old = fixture()
    return ScaledExecutionEnvironment(old.profile, old.cfg, old.target, old.diagnostics,
                                     old.state, old.formal, old.returns, old.tradable,
                                     old.module_map, old.initial)


class Contracts(unittest.TestCase):
    def test_soft_targets_preserve_rank_and_common_offset(self):
        x = np.array([[1., 2., 3., 4.], [0., 0., 0., 0.]])
        p = soft_targets(x, .2)
        np.testing.assert_allclose(p.sum(1), 1.)
        np.testing.assert_allclose(p, soft_targets(x+17., .2))
        self.assertEqual(p[0].argmax(), 3)
        np.testing.assert_allclose(p[1], .25)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            standardizer([[np.nan]])
        with self.assertRaises(ValueError):
            soft_targets([[0., 0., 0., np.nan]], .2)

    def test_future_mutation_does_not_change_normalization(self):
        a, b = environment(), environment()
        b._state[40:] = 1000
        b._returns[40:] = .5
        b._target[40:] = .1
        for x, y in zip(training_normalizer(a, np.arange(30)), training_normalizer(b, np.arange(30))):
            np.testing.assert_array_equal(x, y)

    def test_constant_availability_is_not_confidence(self):
        env = environment()
        mean, std = training_normalizer(env, np.arange(30))
        self.assertTrue(np.isfinite(std).all())
        self.assertTrue((std > 0).all())
        obs = env.observation(0, env.initial_weights(0), env.initial_context(), mean, std)
        self.assertTrue(np.isfinite(obs).all())
        with self.assertRaises(ValueError):
            env.observation(0, env.initial_weights(0), env.initial_context(), mean[:2], std[:2])

    def test_teacher_matured_only_and_suffix_invariant(self):
        env = environment()
        mean, std = training_normalizer(env, np.arange(12))
        before = teacher_data(env, np.arange(12), mean, std, "scaled_value")
        self.assertEqual(len(before[3]), 8)
        self.assertTrue(all(r["label_end_date"] <= r["train_last_execution"] for r in before[3]))
        other = environment()
        other._returns[14:] = 100.
        after = teacher_data(other, np.arange(12), mean, std, "scaled_value")
        torch.testing.assert_close(before[1], after[1])

    def test_mechanical_first_action_has_zero_relative_value(self):
        env = environment()
        context = env.initial_context()
        current = env.initial_weights(4)
        values = env.action_values(4, current, context)
        self.assertAlmostEqual(values[env.mechanical_action(4, current, context)], 0.)
        self.assertEqual(context.age, 0)

    def test_action_logging_separates_minimum_hold(self):
        env = environment()
        info = env.step(0, 3, env.initial_weights(0), env.initial_context())[-1]
        self.assertEqual(info["raw_policy_action"], 3)
        self.assertEqual(info["allowed_policy_action"], 0)
        self.assertTrue(info["minimum_hold_suppressed"])

    def test_one_epoch_real_ppo_update_and_restore(self):
        from src import v45_portfolio as engine
        env = environment()
        env.cfg = copy.deepcopy(env.cfg)
        env.cfg["controller"].update(max_epochs=1, teacher_pretrain_epochs=1, updates_per_epoch=1)
        env.cfg["v48"] = {"variant": "scaled_value"}
        record, logs = fit_seed(env, 2022, 17, 1e-4, .004,
                                (np.arange(12), np.arange(20, 27), np.arange(40, 47)))
        model = engine.restore_model(record, env.cfg["controller"]["hidden_dim"])
        rows = engine.simulate(env, np.arange(40, 47), model, record["mean"], record["std"])
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(rows), 7)


if __name__ == "__main__":
    unittest.main()
