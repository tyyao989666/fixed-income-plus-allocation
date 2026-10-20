from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.v55_learning_objective import temporal_opportunity_targets,decode_opportunity


class ObjectiveTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(17)
        return rng.normal(0,.02,(500,3,3)),np.ones((500,3,3),dtype=bool),rng.normal(0,.01,(500,3))

    def test_inverse(self):
        y,m,r=self.fixture();z=temporal_opportunity_targets(y,m,r);good=z["mask"]
        np.testing.assert_allclose(decode_opportunity(z["target"],z["scale"],z["offset"])[good],y[good],atol=1e-7)

    def test_future_labels_not_used_in_current_offset(self):
        y,m,r=self.fixture();before=temporal_opportunity_targets(y,m,r)
        y[250:]=1e6
        after=temporal_opportunity_targets(y,m,r)
        np.testing.assert_array_equal(before["offset"][:251],after["offset"][:251])

    def test_unmatured_labels_not_used(self):
        y,m,r=self.fixture();before=temporal_opportunity_targets(y,m,r)
        y[300-60,2]=1e6
        after=temporal_opportunity_targets(y,m,r)
        np.testing.assert_array_equal(before["offset"][:301,2],after["offset"][:301,2])
        self.assertFalse(np.array_equal(before["offset"][301,2],after["offset"][301,2]))

    def test_current_return_not_used_in_risk_scale(self):
        y,m,r=self.fixture();before=temporal_opportunity_targets(y,m,r)
        r[250:]=1e6
        after=temporal_opportunity_targets(y,m,r)
        np.testing.assert_array_equal(before["scale"][:251],after["scale"][:251])

    def test_missing_cells_preserved(self):
        y,m,r=self.fixture();m[300,1,1]=False
        z=temporal_opportunity_targets(y,m,r)
        self.assertFalse(z["mask"][300,1,1]);self.assertEqual(z["target"][300,1,1],0)
        self.assertTrue(z["mask"][300,1,0])

    def test_warmup_not_backfilled(self):
        y,m,r=self.fixture();z=temporal_opportunity_targets(y,m,r)
        self.assertFalse(z["mask"][:126].any())


if __name__=="__main__":unittest.main()
