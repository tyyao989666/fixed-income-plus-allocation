import unittest
from src.v68_terminal_guard import terminal_guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.event = dict(ts_code='128032.SZ', conservative_available_after_close='2021-07-02',
            record_date='2021-07-27', redemption_date='2021-07-28', payment_date='2021-08-04',
            gross_cash_yuan_per_bond=100.88, result_publication_date='2021-08-05')

    def test_not_visible_on_availability_close(self):
        self.assertTrue(terminal_guard(self.event,'2021-07-02',1)['buy_allowed'])

    def test_known_schedule_exits_not_flips(self):
        state = terminal_guard(self.event,'2021-07-05',1)
        self.assertFalse(state['buy_allowed'])
        self.assertTrue(state['exit_required'])

    def test_month_end_claim_not_cash(self):
        state = terminal_guard(self.event,'2021-07-31',2, self.event)
        self.assertEqual(state['expected_cash'],201.76)
        self.assertFalse(state['usable_cash_confirmed'])
        self.assertEqual(state['cash_receipt'],0)

    def test_result_not_visible_on_its_date(self):
        self.assertFalse(terminal_guard(self.event,'2021-08-05',1,self.event)['usable_cash_confirmed'])
        self.assertTrue(terminal_guard(self.event,'2021-08-06',1,self.event)['usable_cash_confirmed'])

    def test_no_repeat_cash_posting(self):
        self.assertEqual(terminal_guard(self.event,'2021-08-09',1,self.event)['cash_receipt'],0)

    def test_wrong_security_rejected(self):
        other = dict(self.event, ts_code='wrong')
        with self.assertRaises(ValueError):
            terminal_guard(self.event,'2021-08-09',1,other)


if __name__=='__main__':
    unittest.main()
