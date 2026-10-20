"""Contract tests for the unpromoted v46 experiment."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import v45_portfolio as engine
from src.v46_shadow_research import RelativeExecutionEnvironment, shadow_predictions


def sample():
    return pd.DataFrame([dict(as_of_date=pd.Timestamp("2024-01-02"),
        train_end_date=pd.Timestamp("2022-03-01"), gate_as_of_date=pd.Timestamp("2023-09-01"),
        is_oof=True, selected_candidate="F1", model="F1", seed=-1, task="sleeve",
        entity="equity", horizon_days=20, raw_increment=.02, prediction=.03,
        baseline_prediction=.01, entity_training_dates=300, gate_reason="gate_rejected")])


class ShadowTests(unittest.TestCase):
    def test_keeps_fixed_horizon_weight(self):
        result, status = shadow_predictions(sample())
        self.assertAlmostEqual(result.increment_21.iloc[0], .02*.5*21/20*.25)
        self.assertTrue(status["formal_forecast_gate_bypassed_for_research_only"])

    def test_thin_entity_excluded(self):
        data = sample()
        data["entity_training_dates"] = 299
        self.assertTrue(shadow_predictions(data)[0].empty)

    def test_current_missing_excluded_even_if_prediction_finite(self):
        data = sample()
        data["gate_reason"] = "missing_entity_forecast"
        self.assertTrue(shadow_predictions(data)[0].empty)

    def test_future_training_rejected(self):
        data = sample()
        data["train_end_date"] = data.as_of_date
        with self.assertRaises(ValueError):
            shadow_predictions(data)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            shadow_predictions(pd.concat([sample(), sample()]))

    def test_future_suffix_does_not_change_past(self):
        first, future = sample(), sample()
        future["as_of_date"] = pd.Timestamp("2025-01-02")
        data = pd.concat([first, future])
        before = shadow_predictions(data)[0]
        data.loc[data.as_of_date.dt.year.eq(2025), ["prediction", "raw_increment"]] += 100
        after = shadow_predictions(data)[0]
        pd.testing.assert_frame_equal(before.iloc[:1], after.iloc[:1])

    def test_symmetric_mechanical_reward_is_zero(self):
        env = object.__new__(RelativeExecutionEnvironment)
        env._returns = np.zeros((2, 2))
        env._anchor = np.array([[.4, .6]])
        env.cfg = {"controller": {"active_weight_penalty": .002}}
        context = SimpleNamespace(formal_weights=np.array([.5, .5]))
        info = dict(reward=-.04, reward_active_penalty=.04, active_l1=.2,
                    reward_return_edge=0., reward_downside_penalty=0.,
                    reward_volatility_penalty=0., reward_drawdown_penalty=0.)
        with patch.object(engine.ExecutionEnvironment, "step", return_value=(-.04, 1, np.array([.5, .5]), context, info)):
            result = env.step(0, 0, np.array([.5, .5]), None)
        self.assertAlmostEqual(result[0], 0., places=10)


if __name__ == "__main__":
    unittest.main()
