"""Isolated single-position terminal-event ledger, not a certified total-return index."""
import math
import pandas as pd


class TerminalPositionBook:
    def __init__(self, quantity, opening_price, opening_date, schedule, result=None, fee_rate=.003):
        if not all(math.isfinite(v) for v in [quantity, opening_price, fee_rate]) or quantity < 0 or opening_price <= 0 or not 0 <= fee_rate < 1:
            raise ValueError('Invalid position/cost')
        self.schedule = dict(schedule)
        self.result = dict(result) if result is not None else None
        self.last_day = pd.Timestamp(opening_date)
        self.quantity, self.last_mark = float(quantity), float(opening_price)
        self.cash, self.claim, self.cost = 0., 0., 0.
        self.entitlement = None
        self.cancelled = self.settled = False
        self.fee_rate = fee_rate
        self.available = pd.Timestamp(schedule['conservative_available_after_close'])
        self.record = pd.Timestamp(schedule['record_date'])
        self.ex_date = pd.Timestamp(schedule['redemption_date'])
        self.pay_date = pd.Timestamp(schedule['payment_date'])
        self.last_trade = pd.Timestamp(schedule.get('last_trade_date', schedule['record_date']))
        self.amount = float(schedule['gross_cash_yuan_per_bond'])
        if not math.isfinite(self.amount) or self.amount <= 0 or not self.available <= self.record < self.ex_date <= self.pay_date or self.last_trade > self.record:
            raise ValueError('Invalid schedule chronology/amount')
        if self.last_day > self.record:
            raise ValueError('Opening entitlement requires a pre-record position')
        if self.last_day == self.record:
            self.entitlement = self.quantity
        if self.result is not None:
            if self.result['ts_code'] != schedule['ts_code'] or pd.Timestamp(self.result['payment_date']) != self.pay_date:
                raise ValueError('Result identity/payment mismatch')
            if not math.isclose(float(self.result['gross_cash_yuan_per_bond']), self.amount, rel_tol=0, abs_tol=1e-9):
                raise ValueError('Result amount mismatch')

    def step(self, day, price=None, volume=None, exit_on_known_schedule=False):
        day = pd.Timestamp(day)
        if day <= self.last_day:
            raise ValueError('Dates must increase; duplicate processing forbidden')
        if self.last_day < self.record < day and self.entitlement is None:
            raise ValueError('Record date skipped; entitlement not established')
        known = day > self.available
        mark_known = price is not None and math.isfinite(float(price)) and price > 0
        executable = mark_known and volume is not None and math.isfinite(float(volume)) and volume > 0
        exit_requested = bool(exit_on_known_schedule and known and self.quantity > 0 and not self.cancelled)
        fill_qty = cost_today = 0.
        if exit_requested and executable and day <= self.last_trade:
            fill_qty = self.quantity
            cost_today = fill_qty*price*self.fee_rate
            self.cash += fill_qty*price-cost_today
            self.cost += cost_today
            self.quantity = 0.
        if day == self.record:
            self.entitlement = self.quantity
        if day >= self.ex_date and known and not self.cancelled:
            if self.entitlement is None:
                raise ValueError('No record-date entitlement')
            if not math.isclose(self.quantity, self.entitlement, abs_tol=1e-12):
                raise ValueError('Inventory no longer matches registered entitlement')
            self.claim = self.entitlement*self.amount
            self.quantity = 0.
            self.cancelled = True
        result_known = self.result is not None and day > pd.Timestamp(self.result['result_publication_date'])
        receipt = 0.
        if result_known and day >= self.pay_date and not self.settled:
            if not self.cancelled:
                raise ValueError('Payment before cancellation')
            receipt = self.claim
            self.cash += receipt
            self.claim = 0.
            self.settled = True
        missing_mark = self.quantity > 0 and not mark_known
        if mark_known:
            self.last_mark = float(price)
        nav = self.cash+self.claim+self.quantity*self.last_mark
        self.last_day = day
        return dict(date=str(day.date()), quantity=self.quantity, cash=self.cash, receivable=self.claim,
            provisional_nav=nav, observed_mark_nav=None if missing_mark else nav,
            missing_held_mark=missing_mark, schedule_known=known, result_known=bool(result_known),
            exit_requested=exit_requested, sale_quantity=fill_qty, cost=cost_today,
            receipt=receipt, cancelled=self.cancelled, settled=self.settled,
            formal_total_return_certified=False)
