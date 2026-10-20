import unittest
from src.v69_terminal_position_book import TerminalPositionBook


class BookTests(unittest.TestCase):
    def setUp(self):
        self.schedule = dict(ts_code='X', conservative_available_after_close='2021-07-02',
            record_date='2021-07-27', redemption_date='2021-07-28', payment_date='2021-08-04',
            gross_cash_yuan_per_bond=100.88)
        self.result = dict(self.schedule, result_publication_date='2021-08-05')

    def book(self, result=True):
        return TerminalPositionBook(2, 157.35, '2021-06-30', self.schedule, self.result if result else None)

    def test_receivable_then_payment_once(self):
        b = self.book()
        b.step('2021-07-27', 170, 10)
        z = b.step('2021-07-28')
        self.assertEqual(z['quantity'], 0)
        self.assertEqual(z['receivable'], 201.76)
        self.assertEqual(z['cash'], 0)
        self.assertEqual(b.step('2021-08-04')['cash'], 0)
        self.assertEqual(b.step('2021-08-05')['cash'], 0)
        paid = b.step('2021-08-06')
        self.assertEqual(paid['receipt'], 201.76)
        self.assertEqual(paid['provisional_nav'], z['provisional_nav'])
        self.assertEqual(b.step('2021-08-09')['receipt'], 0)

    def test_sale_and_settlement_do_not_double_count(self):
        b = self.book()
        sale = b.step('2021-07-05',160.52,100,True)
        self.assertAlmostEqual(sale['cash'],2*160.52*(1-.003))
        b.step('2021-07-27')
        b.step('2021-07-28')
        self.assertEqual(b.step('2021-08-06')['receipt'],0)

    def test_unknown_same_close_cannot_trigger_exit(self):
        self.assertEqual(self.book().step('2021-07-02',160,100,True)['sale_quantity'],0)

    def test_zero_volume_cannot_sell(self):
        b=self.book()
        self.assertEqual(b.step('2021-07-05',160,0,True)['sale_quantity'],0)

    def test_no_result_no_fabricated_cash(self):
        b=self.book(False)
        b.step('2021-07-27',170,10)
        z=b.step('2021-09-01')
        self.assertEqual(z['cash'],0)
        self.assertEqual(z['receivable'],201.76)

    def test_skipped_record_date_blocks(self):
        with self.assertRaises(ValueError):
            self.book().step('2021-07-28')

    def test_missing_price_not_strict_valuation(self):
        z=self.book().step('2021-07-05')
        self.assertIsNone(z['observed_mark_nav'])
        self.assertTrue(z['missing_held_mark'])

    def test_future_result_does_not_change_past(self):
        a,b=self.book(),self.book(False)
        self.assertEqual(a.step('2021-07-05',160,10), b.step('2021-07-05',160,10))

    def test_wrong_payment_amount_rejected(self):
        result=dict(self.result,gross_cash_yuan_per_bond=200)
        with self.assertRaises(ValueError):
            TerminalPositionBook(2,157,'2021-06-30',self.schedule,result)

    def test_duplicate_day_rejected(self):
        b=self.book()
        b.step('2021-07-05',160,10)
        with self.assertRaises(ValueError):
            b.step('2021-07-05',160,10)


if __name__=='__main__':
    unittest.main()
