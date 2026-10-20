import copy
import unittest
from scripts.replay_v86_inventory_maturities import CASES, exact_total


class MaturityEvidenceTests(unittest.TestCase):
    def test_all_three_contract_totals_reconcile_exactly(self):
        for case in CASES:
            with self.subTest(code=case["code"]):
                self.assertTrue(exact_total(case))

    def test_small_discrepancy_is_not_silently_tolerated(self):
        case = copy.deepcopy(CASES[0])
        case["aggregate"] = "1430735098.01"
        with self.assertRaises(ValueError):
            exact_total(case)


if __name__ == "__main__":
    unittest.main()
