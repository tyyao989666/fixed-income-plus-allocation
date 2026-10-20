"""Quantity ledger for isolated CB execution research, not a certified wealth index."""
import numpy as np
import pandas as pd


EVENT_COLUMNS = ["event_id", "ts_code", "kind", "known_at", "record_date",
                 "ex_date", "pay_date", "confirmed_at", "cash_yuan_per_bond", "source"]


def validate_events(frame, codes):
    if not set(EVENT_COLUMNS).issubset(frame):
        raise ValueError("Missing cashflow contract fields")
    f = frame[EVENT_COLUMNS].copy()
    if f.empty:
        return f
    if f.event_id.isna().any() or f.event_id.duplicated().any():
        raise ValueError("Cashflow events must have unique identifiers")
    if not f.ts_code.isin(codes).all() or not f.kind.isin(["coupon", "terminal"]).all():
        raise ValueError("Invalid event identity/kind")
    for c in ["known_at", "record_date", "ex_date", "pay_date", "confirmed_at"]:
        f[c] = pd.to_datetime(f[c], errors="raise")
        if f[c].isna().any():
            raise ValueError("Cashflow timestamps cannot be missing")
    amount = pd.to_numeric(f.cash_yuan_per_bond, errors="raise")
    if not np.isfinite(amount).all() or amount.lt(0).any():
        raise ValueError("Invalid cashflow in yuan per bond")
    f["cash_yuan_per_bond"] = amount
    if not ((f.known_at <= f.record_date) & (f.record_date < f.ex_date)
            & (f.ex_date <= f.pay_date) & (f.confirmed_at <= f.pay_date)).all():
        raise ValueError("Cashflow chronology is incomplete or look-ahead")
    if f.source.isna().any() or f.source.astype(str).str.strip().eq("").any():
        raise ValueError("Every cashflow requires source evidence")
    # Distinct vendor identifiers must not duplicate the same economic payment.
    if f.duplicated(["ts_code", "record_date", "ex_date", "pay_date"]).any():
        raise ValueError("Duplicate economic cashflow")
    return f


class QuantityBooks:
    """Vectorized close execution; no price interpolation or phantom liquidation."""

    def __init__(self, names, codes, initial_capital=1_000_000., cost_rate=.0005):
        if (len(set(names)) != len(names) or len(set(codes)) != len(codes)
                or not np.isfinite(initial_capital) or initial_capital <= 0
                or not 0 <= cost_rate < 1):
            raise ValueError("Invalid ledger contract")
        self.names, self.codes = list(names), list(codes)
        self.quantity = np.zeros((len(names), len(codes)))
        self.last_mark = np.full(len(codes), np.nan)
        self.cash = np.full(len(names), initial_capital)
        self.cost_rate = cost_rate
        self.receivables = {}
        self.entitlements = {}
        self.recorded = set()
        self.accrued = set()
        self.paid = set()
        self.last_day = None

    def step(self, day, target, buy_allowed, price, executable, valuation, events):
        """Targets/buy permission must be fixed at the previous close by caller.

        Events are evidence-adjudicated inputs, not forecasts of scheduled payments.
        Initial purchase occurs at this close, hence earns no pre-purchase return.
        """
        day = pd.Timestamp(day)
        if self.last_day is not None and day <= self.last_day:
            raise ValueError("Ledger dates must increase; cannot pay twice")
        target = np.asarray(target, dtype=float)
        buy_allowed = np.asarray(buy_allowed, dtype=bool)
        price, valuation = np.asarray(price, dtype=float), np.asarray(valuation, dtype=float)
        executable = np.asarray(executable, dtype=bool)
        n = len(self.codes)
        if target.shape != self.quantity.shape or any(x.shape != (n,) for x in [buy_allowed, price, valuation, executable]):
            raise ValueError("Ledger shape mismatch")
        if not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("Invalid target")
        total = target.sum(axis=1)
        if not (np.isclose(total, 0, atol=1e-10) | np.isclose(total, 1, atol=1e-10)).all():
            raise ValueError("Target must be a simplex or unavailable (all zero)")
        event_cash = np.zeros(len(self.names))
        for row in events.itertuples(index=False):
            event, code = row.event_id, self.codes.index(row.ts_code)
            if row.ex_date <= day and event not in self.accrued:
                if event not in self.entitlements:
                    # Events whose record precedes portfolio inception convey no rights.
                    if self.last_day is not None and row.record_date >= self.last_day:
                        raise ValueError("Missing record-date entitlement")
                    self.entitlements[event] = np.zeros(len(self.names))
                if row.kind == "terminal":
                    if not np.allclose(self.quantity[:, code], self.entitlements[event]):
                        raise ValueError("Terminal cancellation does not match registered inventory")
                    self.quantity[:, code] = 0
                self.receivables[event] = self.entitlements[event] * row.cash_yuan_per_bond
                self.accrued.add(event)
            if row.pay_date <= day and event not in self.paid:
                amount = self.receivables.pop(event)
                self.cash += amount
                event_cash += amount
                self.paid.add(event)
        retired = np.zeros(n, dtype=bool)
        for row in events.itertuples(index=False):
            if row.kind == "terminal" and row.ex_date <= day:
                retired[self.codes.index(row.ts_code)] = True
        tradable = executable & np.isfinite(price) & (price > 0) & ~retired
        observed = np.isfinite(valuation) & (valuation > 0)
        marks = np.where(tradable, price, np.where(observed, valuation, self.last_mark))
        held = self.quantity > 1e-12
        missing = held & ~(tradable | observed)[None, :]
        if (held & (~np.isfinite(marks) | (marks <= 0))[None, :]).any():
            raise ValueError("Held quantities lack even a provisional mark")
        receivable = sum(self.receivables.values(), start=np.zeros(len(self.names)))
        market = self.quantity * np.nan_to_num(marks, nan=0.)
        before = market.sum(axis=1) + self.cash + receivable
        if (before <= 0).any() or not np.isfinite(before).all():
            raise ValueError("Non-positive ledger capital")
        desired = np.divide(target * before[:, None], marks[None, :],
                            out=self.quantity.copy(), where=np.isfinite(marks[None, :]) & (marks[None, :] > 0))
        requested = desired - self.quantity
        # Unknown marks prevent NAV-sized reallocation, but a prior zero target
        # still permits a full exit from OTHER securities with actual execution.
        active = np.isclose(total, 1, atol=1e-10) & ~missing.any(axis=1)
        known_exit = np.isclose(total, 1, atol=1e-10)[:, None] & (target == 0)
        sell_request = np.where(active[:, None], np.minimum(requested, 0),
                                np.where(known_exit, -self.quantity, 0))
        sells = np.where(tradable[None, :], sell_request, 0)
        unit_price = np.where(tradable, price, 0)
        sell_value = (-sells * unit_price).sum(axis=1)
        self.cash += sell_value * (1 - self.cost_rate)
        self.quantity += sells
        buys = np.where((tradable & buy_allowed)[None, :] & active[:, None], np.maximum(requested, 0), 0)
        need = (buys * unit_price).sum(axis=1) * (1 + self.cost_rate)
        scale = np.minimum(1., np.divide(np.maximum(self.cash, 0), need, out=np.zeros_like(need), where=need > 0))
        buys *= scale[:, None]
        buy_value = (buys * unit_price).sum(axis=1)
        self.cash -= buy_value * (1 + self.cost_rate)
        self.quantity += buys
        if (self.cash < -1e-7).any() or (self.quantity < -1e-12).any():
            raise ValueError("Ledger borrowing/short sale")
        cost = (sell_value + buy_value) * self.cost_rate
        nav = (self.quantity * np.nan_to_num(marks, nan=0.)).sum(axis=1) + self.cash + receivable
        if not np.allclose(nav, before - cost, rtol=1e-10, atol=1e-7):
            raise ValueError("Execution failed NAV/cost conservation")
        for row in events.itertuples(index=False):
            if row.record_date == day and row.event_id not in self.recorded:
                self.entitlements[row.event_id] = self.quantity[:, self.codes.index(row.ts_code)].copy()
                self.recorded.add(row.event_id)
            elif row.record_date < day and row.event_id not in self.recorded and self.last_day is not None and row.record_date > self.last_day:
                raise ValueError("Record date skipped; calendar entitlement needs adjudication")
        self.last_mark, self.last_day = marks.copy(), day
        return dict(provisional_nav=nav, observed_price_nav=np.where(missing.any(axis=1), np.nan, nav),
            cost_yuan=cost, traded_notional=buy_value + sell_value, cash=self.cash.copy(),
            receivable=receivable, payment_cash=event_cash, missing_held_count=missing.sum(axis=1),
            frozen_held_count=(held & ~tradable[None, :]).sum(axis=1),
            fills=sells + buys, stale_valuation_positions=missing,
            blocked_requested_notional=(np.abs(requested - sells - buys) * np.nan_to_num(marks, nan=0.)).sum(axis=1))
