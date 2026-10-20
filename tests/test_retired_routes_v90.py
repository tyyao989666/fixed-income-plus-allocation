import unittest

from src.v90_retired_security_routes import route_at_signal, conditional_cash_reference


def route():
    return dict(old_code="123015.SZ", new_code="404001", known_at="2023-09-15",
                transfer_start="2023-09-18", scheduled_weekdays=[4], source_sha256="a" * 64)


class RouteTests(unittest.TestCase):
    def test_before_transfer_not_routed(self):
        self.assertFalse(route_at_signal("123015.SZ", "2023-09-15", [route()])["routed"])

    def test_started_and_visible(self):
        out = route_at_signal("123015.SZ", "2023-09-18", [route()])
        self.assertEqual(out["query_code"], "404001")
        self.assertFalse(out["execution_authorized"])
        self.assertFalse(out["rights_confirmed"])

    def test_future_knowledge_hidden(self):
        row = route()
        row["known_at"] = "2024-01-01"
        self.assertFalse(route_at_signal("123015.SZ", "2023-09-18", [row])["routed"])

    def test_future_append_does_not_change_past(self):
        row = route()
        row["known_at"] = "2025-01-01"
        self.assertEqual(route_at_signal("123015.SZ", "2024-01-01", [route()]),
                         route_at_signal("123015.SZ", "2024-01-01", [route(), row]))

    def test_ambiguous_revisions_rejected(self):
        with self.assertRaises(ValueError):
            route_at_signal("123015.SZ", "2024-01-01", [route(), route()])

    def test_bad_evidence_rejected(self):
        row = route()
        row["source_sha256"] = ""
        with self.assertRaises(ValueError):
            route_at_signal("123015.SZ", "2024-01-01", [row])

    def test_reference_not_backdated_to_payment(self):
        out = conditional_cash_reference(100, 100.14, "2024-01-31", "2025-04-28", assumptions_explicit=True)
        self.assertIsNone(out["scenario_gross_cash_reference_yuan"])
        out = conditional_cash_reference(100, 100.14, "2025-04-28", "2025-04-28", assumptions_explicit=True)
        self.assertEqual(out["scenario_gross_cash_reference_yuan"], 10014)
        self.assertFalse(out["posting_allowed"])
        self.assertIsNone(out["actual_cash_received_yuan"])

    def test_missing_assumptions_rejected(self):
        with self.assertRaises(ValueError):
            conditional_cash_reference(100, 100, "2025-04-28", "2025-04-28", assumptions_explicit=False)

    def test_unknown_date_or_amount_rejected(self):
        for date, qty in [(None, 100), ("2025-04-28", float("nan"))]:
            with self.assertRaises(ValueError):
                conditional_cash_reference(qty, 100, date, "2025-04-28", assumptions_explicit=True)


if __name__ == "__main__":
    unittest.main()
