import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.diagnose_v59_signal import feature_similarity


class SimilarityTests(unittest.TestCase):
    def test_rotation_scale_and_translation(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(50, 6))
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        self.assertAlmostEqual(feature_similarity(x, x), 1.)
        self.assertAlmostEqual(feature_similarity(x, 3*x@q+7), 1.)

    def test_constant_has_no_similarity_claim(self):
        self.assertIsNone(feature_similarity(np.ones((5, 3)), np.ones((5, 3))))

    def test_alignment_rejected(self):
        self.assertIsNone(feature_similarity(np.ones((5, 3)), np.ones((6, 3))))


if __name__ == "__main__":
    unittest.main()
