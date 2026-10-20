"""Event adapter for CB quantity books: disclose first, settle only when confirmed."""
import copy

import numpy as np
import pandas as pd

from src.v52_cb_execution import QuantityBooks, EVENT_COLUMNS


def validate_causal_events(frame, codes):
    if set(EVENT_COLUMNS) - set(frame):
        raise ValueError("Missing canonical cashflow fields")
    f = frame.copy()
    if f.empty:
        return f
    if (f.event_id.isna().any() or f.event_id.duplicated().any() or not f.ts_code.isin(codes).all()
            or not f.kind.isin(["terminal", "coupon"]).all()):
        raise ValueError("Ambiguous event identity or kind")
    for name in ("known_at", "record_date", "ex_date", "pay_date", "confirmed_at"):
        f[name] = pd.to_datetime(f[name], errors="raise")
        if name != "confirmed_at" and f[name].isna().any():
            raise ValueError("Unknown schedule chronology")
    if not ((f.known_at <= f.record_date) & (f.record_date < f.ex_date) & (f.ex_date <= f.pay_date)).all():
        raise ValueError("Schedule chronology invalid")
    confirmed = f.confirmed_at.notna()
    if not f.loc[confirmed, "confirmed_at"].ge(f.loc[confirmed, "known_at"]).all():
        raise ValueError("Confirmation cannot predate schedule availability")
    f.cash_yuan_per_bond = pd.to_numeric(f.cash_yuan_per_bond, errors="raise")
    if (not np.isfinite(f.cash_yuan_per_bond).all() or f.cash_yuan_per_bond.lt(0).any()
            or f.source.isna().any() or f.source.astype(str).str.strip().eq("").any()
            or f.duplicated(["ts_code", "record_date", "ex_date", "pay_date"]).any()):
        raise ValueError("Invalid amount, missing source or duplicate economic payment")
    # A result published AFTER the scheduled payment is valid evidence, not bad data.
    return f


class CausalQuantityBooks(QuantityBooks):
    def __init__(self, names, codes, events, sessions, **kwargs):
        super().__init__(names, codes, **kwargs)
        self.events = validate_causal_events(events, codes)
        self.sessions = pd.DatetimeIndex(sessions)
        if self.sessions.has_duplicates or not self.sessions.is_monotonic_increasing:
            raise ValueError("Explicit ordered domestic calendar required")
        self.inception = None

    def step(self, day, target, buy_allowed, price, executable, valuation, *, signal_date):
        snapshot = copy.deepcopy(self.__dict__)
        try:
            return self._causal_step(day, target, buy_allowed, price, executable, valuation, signal_date)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise

    def _causal_step(self, day, target, buy_allowed, price, executable, valuation, signal_date):
        day, signal = pd.Timestamp(day), pd.Timestamp(signal_date)
        if day not in self.sessions or (self.last_day is not None and day <= self.last_day):
            raise ValueError("Execution must follow ordered domestic sessions")
        i = self.sessions.get_loc(day)
        if i == 0 or self.sessions[i - 1] != signal or (self.last_day is not None and signal != self.last_day):
            raise ValueError("Target must come from exactly the previous domestic close")
        if self.inception is None:
            self.inception = day
        visible = self.events.loc[self.events.known_at <= day]
        retired = np.zeros(len(self.codes), dtype=bool)
        payment = np.zeros(len(self.names))
        transitions = []
        for event in visible.itertuples(index=False):
            code = self.codes.index(event.ts_code)
            if event.record_date < day and event.event_id not in self.entitlements:
                if event.record_date < self.inception:
                    self.entitlements[event.event_id] = np.zeros(len(self.names))
                    self.recorded.add(event.event_id)
                else:
                    raise ValueError("Missing actual record-date inventory")
            if event.ex_date <= day and event.event_id not in self.accrued:
                if event.event_id not in self.entitlements:
                    raise ValueError("No registered entitlement for ex-date")
                owned = self.entitlements[event.event_id]
                if event.kind == "terminal":
                    if not np.allclose(self.quantity[:, code], owned, atol=1e-10, rtol=1e-10):
                        raise ValueError("Terminal cancellation differs from registered inventory")
                    self.quantity[:, code] = 0
                claim = owned * event.cash_yuan_per_bond
                self.receivables[event.event_id] = claim
                self.accrued.add(event.event_id)
                transitions.append(dict(event_id=event.event_id, ts_code=event.ts_code, day=day,
                    transition="entitlement_to_receivable", amount_all_books=float(claim.sum())))
            if (event.pay_date <= day and pd.notna(event.confirmed_at) and event.confirmed_at <= day
                    and event.event_id not in self.paid):
                if event.event_id not in self.receivables:
                    raise ValueError("Payment without accrued receivable")
                amount = self.receivables.pop(event.event_id)
                self.cash += amount
                payment += amount
                self.paid.add(event.event_id)
                transitions.append(dict(event_id=event.event_id, ts_code=event.ts_code, day=day,
                    transition="confirmed_receivable_to_cash", amount_all_books=float(amount.sum()),
                    confirmed_at=event.confirmed_at, scheduled_pay_date=event.pay_date))
            if event.kind == "terminal" and event.ex_date <= day:
                retired[code] = True
        executable = np.asarray(executable, dtype=bool) & ~retired
        buy_allowed = np.asarray(buy_allowed, dtype=bool) & ~retired
        # The parent handles only fills/costs; no event may bypass the visibility adapter.
        result = super().step(day, target, buy_allowed, price, executable, valuation,
                              pd.DataFrame(columns=EVENT_COLUMNS))
        for event in visible.itertuples(index=False):
            if event.record_date == day and event.event_id not in self.recorded:
                self.entitlements[event.event_id] = self.quantity[:, self.codes.index(event.ts_code)].copy()
                self.recorded.add(event.event_id)
        result["payment_cash"] = payment
        result["event_transitions"] = transitions
        return result
