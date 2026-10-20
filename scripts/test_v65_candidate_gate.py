from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.v65_candidate_gate import zero_or_neural


class CandidateGateTests(unittest.TestCase):
    def test_no_gain_uses_zero(self):
        for ratio in [1., 1.01, .99999]:
            self.assertFalse(zero_or_neural(ratio, 1, 126)['neural_enabled'])

    def test_improvement_is_research_only(self):
        result = zero_or_neural(.98, 15, 126)
        self.assertTrue(result['neural_enabled'])
        self.assertFalse(result['formal_promotion'])

    def test_missing_not_promoted(self):
        self.assertFalse(zero_or_neural(np.nan, 1, 126)['neural_enabled'])
        self.assertFalse(zero_or_neural(.9, 1, 10)['neural_enabled'])
        self.assertFalse(zero_or_neural(.9, 0, 126)['neural_enabled'])
        self.assertFalse(zero_or_neural(.9, np.nan, 126)['neural_enabled'])
        self.assertFalse(zero_or_neural(.9, 1, np.nan)['neural_enabled'])


if __name__ == '__main__':
    unittest.main()
