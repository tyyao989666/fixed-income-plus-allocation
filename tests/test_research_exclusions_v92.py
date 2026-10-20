import unittest

import pandas as pd

from src.v92_cb_research_universe import apply_exclusions


class ExclusionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"A": [True, False], "B": [True, True], "C": [False, True]},
                            index=pd.date_range("2020-01-01", periods=2))

    def test_mask_before_allocation_not_delete_rows(self):
        original = self.frame()
        result = apply_exclusions(original, ["B"])
        self.assertFalse(result.B.any())
        self.assertTrue(original.B.all())
        pd.testing.assert_frame_equal(result[["A", "C"]], original[["A", "C"]])
        self.assertTrue(result.index.equals(original.index))
        self.assertTrue(result.columns.equals(original.columns))

    def test_unknown_code_rejected(self):
        with self.assertRaises(ValueError):
            apply_exclusions(self.frame(), ["missing"])

    def test_duplicate_code_rejected(self):
        with self.assertRaises(ValueError):
            apply_exclusions(self.frame(), ["B", "B"])

    def test_unknown_eligibility_rejected(self):
        frame = self.frame().astype("boolean")
        frame.iloc[0, 0] = pd.NA
        with self.assertRaises(ValueError):
            apply_exclusions(frame, ["B"])

    def test_no_codes_is_identity(self):
        pd.testing.assert_frame_equal(self.frame(), apply_exclusions(self.frame(), []))


if __name__ == "__main__":
    unittest.main()
