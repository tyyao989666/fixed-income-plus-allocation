import unittest
from decimal import Decimal

from src.v89_distressed_claims import creditor_tiers


def claim(claim_id, creditor="A", value=80000):
    return dict(plan_id="synthetic_plan", claim_id=claim_id, creditor_id=creditor, accepted_claim_yuan=value,
                scope_verified=True, aggregate_complete=True)


class CreditorTierTests(unittest.TestCase):
    def test_one_cap_not_two_baskets(self):
        out = creditor_tiers([claim("basket1"), claim("basket2")], 100000, 20000000, "8.70")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["contractual_cash_tier_yuan"], Decimal(100000))
        self.assertEqual(out[0]["indicative_share_units"], Decimal(5220))
        self.assertIsNone(out[0]["recovery_value_yuan"])
        self.assertFalse(out[0]["posting_allowed"])

    def test_two_verified_legal_creditors(self):
        out = creditor_tiers([claim("a"), claim("b", "B")], 100000)
        self.assertEqual(sum(x["contractual_cash_tier_yuan"] for x in out), 160000)

    def test_threshold_boundary(self):
        out = creditor_tiers([claim("a", value=50315)], 50315)[0]
        self.assertEqual(out["residual_unpriced_claim_basis_yuan"], 0)
        out = creditor_tiers([claim("a", value="50315.01")], 50315)[0]
        self.assertEqual(out["residual_unpriced_claim_basis_yuan"], Decimal("0.01"))

    def test_missing_scope_fails_closed(self):
        for key in ["plan_id", "creditor_id", "claim_id", "aggregate_complete", "scope_verified"]:
            row = claim("a")
            row.pop(key)
            with self.assertRaises(ValueError):
                creditor_tiers([row], 100000)

    def test_duplicate_claim_rejected(self):
        with self.assertRaises(ValueError):
            creditor_tiers([claim("a"), claim("a")], 100000)

    def test_independent_plans_not_combined(self):
        other = claim("b")
        other["plan_id"] = "other_issuer"
        with self.assertRaises(ValueError):
            creditor_tiers([claim("a"), other], 100000)

    def test_invalid_amounts(self):
        for value in [float("nan"), float("inf"), -1, "invalid"]:
            with self.assertRaises(ValueError):
                creditor_tiers([claim("a", value=value)], 100000)

    def test_large_claim_is_not_stock_market_value(self):
        out = creditor_tiers([claim("a", value=21000000)], 100000, 20000000, "8.70")[0]
        self.assertEqual(out["residual_unpriced_claim_basis_yuan"], 1000000)
        self.assertIsNone(out["cash_received_yuan"])
        self.assertIsNone(out["recovery_value_yuan"])


if __name__ == "__main__":
    unittest.main()
