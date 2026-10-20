import unittest
from src.v84_receipt_check import reconcile_redemption_total


class ReceiptCheckTests(unittest.TestCase):
    def test_geli_exact_total(self):
        self.assertTrue(reconcile_redemption_total(3968820, "106.00", "420694920.00", 2)["aggregate_reconciled"])

    def test_tianma_contradiction_is_not_rounding(self):
        result = reconcile_redemption_total(29670, "100.353", "2978962.25", 3)
        self.assertFalse(result["aggregate_reconciled"])
        self.assertAlmostEqual(result["discrepancy_yuan"], 1488.74)

    def test_genuine_precision_difference_allowed(self):
        self.assertTrue(reconcile_redemption_total(1000, "100.353", "100353.40", 3)["aggregate_reconciled"])

    def test_nonfinite_and_fractional_bond_totals_rejected(self):
        for bonds in ["NaN", "Infinity", -1, 1.5]:
            with self.assertRaises(ValueError):
                reconcile_redemption_total(bonds, 100, 100, 2)


if __name__ == "__main__":
    unittest.main()
