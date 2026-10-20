import unittest
from scripts.replay_v87_inventory_maturities import CASES, exact_total


class BatchMaturityTests(unittest.TestCase):
    def test_five_exact_totals(self):
        for case in CASES:
            with self.subTest(code=case["code"]):
                exact_total(case)

    def test_haiyin_redemption_principal_is_not_par_principal(self):
        case = next(r for r in CASES if r["code"] == "127003.SZ")
        self.assertEqual(case["bonds"] * 100, 89294600)
        self.assertNotEqual(case["bonds"] * 100, 96438168)
        exact_total(case)

    def test_small_total_discrepancy_rejected(self):
        case = dict(CASES[-1], aggregate="1416800.01")
        with self.assertRaises(ValueError):
            exact_total(case)


if __name__ == "__main__":
    unittest.main()
