import json
import unittest

import pandas as pd

from src.v91_off_exchange_quote_contract import review_receipt, default_state_at_signal


ROUTE = dict(old_code="123015.SZ", new_code="404001", known_at="2023-09-15",
             transfer_start="2023-09-18", scheduled_weekdays=[4])


def receipt(day="20230922", volume="", unit="元", code="404001.NQ"):
    answer = f"|证券代码|证券简称|日期|收盘价|成交量|\n|---|---|---|---|---|\n|{code}|蓝盾退债|{day}|25.13|{volume}|"
    data = dict(answer=answer, indicators_params={"成交量": {"单位设置": unit}})
    return dict(start="2023-09-18", end="2023-09-24", response=dict(ok=True,
                data=dict(result=dict(content=[dict(type="text", text=json.dumps(dict(code=1, data=data)))]))))


class QuoteContractTests(unittest.TestCase):
    def review(self, value):
        return review_receipt(value, ROUTE, pd.bdate_range("2023-09-18", "2023-09-22"))

    def test_missing_quantity_not_zero(self):
        out = self.review(receipt()).iloc[0]
        self.assertIsNone(out.observed_volume)
        self.assertIn("missing_quantity", out.reasons)
        self.assertIn("quantity_unit_not_verified", out.reasons)

    def test_weekend_not_trade(self):
        out = self.review(receipt(day="20230923", volume="0")).iloc[0]
        self.assertIn("not_domestic_session", out.reasons)
        self.assertIn("no_positive_quantity", out.reasons)

    def test_wrong_security_rejected_for_execution(self):
        out = self.review(receipt(code="400194.NQ")).iloc[0]
        self.assertIn("security_identity_mismatch", out.reasons)
        self.assertFalse(out.execution_allowed)

    def test_positive_quantity_does_not_certify_otc_execution(self):
        out = self.review(receipt(volume="1000", unit="张")).iloc[0]
        self.assertFalse(out.approved_for_nav)
        self.assertFalse(out.execution_allowed)

    def test_initial_nontransfer_day(self):
        self.assertIn("not_initial_scheduled_transfer_day", self.review(receipt(day="20230921")).iloc[0].reasons)

    def test_unknown_quantity_not_interpolated(self):
        self.assertIsNone(self.review(receipt(volume="--")).iloc[0].observed_volume)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            self.review(receipt(volume="NaN"))

    def test_outside_query_rejected(self):
        with self.assertRaises(ValueError):
            self.review(receipt(day="20231001"))

    def test_default_not_visible_early(self):
        note = dict(old_code="110072.SH", known_at="2026-08-18", source_sha256="a" * 64)
        self.assertFalse(default_state_at_signal("110072.SH", "2026-08-17", [note])["nonpayment_notice_visible"])

    def test_default_is_not_zero_recovery(self):
        note = dict(old_code="110072.SH", known_at="2026-08-18", source_sha256="a" * 64)
        out = default_state_at_signal("110072.SH", "2026-08-18", [note])
        self.assertTrue(out["nonpayment_notice_visible"])
        self.assertIsNone(out["recovery_value_yuan"])
        self.assertFalse(out["auto_par_redemption_allowed"])

    def test_future_default_append_does_not_change_past(self):
        note = dict(old_code="110072.SH", known_at="2026-08-18", source_sha256="a" * 64)
        self.assertEqual(default_state_at_signal("110072.SH", "2024-01-01", []),
                         default_state_at_signal("110072.SH", "2024-01-01", [note]))


if __name__ == "__main__":
    unittest.main()
