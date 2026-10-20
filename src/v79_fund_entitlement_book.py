"""Single-position fund accounting: quantity, record-date entitlement and receivable."""
import math

import pandas as pd


class FundEntitlementBook:
    """A diagnostic book, not a full trading engine or verified opening inventory."""
    def __init__(self, code, opening_date, quantity, opening_price, schedules, splits=None, receipts=None):
        if not all(math.isfinite(float(x)) for x in (quantity, opening_price)) or quantity <= 0 or opening_price <= 0:
            raise ValueError("Positive finite opening position required")
        self.code, self.day = code, pd.Timestamp(opening_date)
        self.quantity, self.mark = float(quantity), float(opening_price)
        self.cash, self.claims = 0., {}
        self.history = {self.day: self.quantity}
        self.schedules = schedules.copy()
        if not schedules.empty and (not schedules.ts_code.eq(code).all() or schedules.event_id.duplicated().any()):
            raise ValueError("Exact code and unique distribution events required")
        self.splits = list(splits or [])
        self.receipts = list(receipts or [])
        self.applied_splits, self.settled = set(), set()

    def step(self, day, price=None):
        fields = ("day", "quantity", "mark", "cash", "claims", "history", "applied_splits", "settled")
        snapshot = {name: getattr(self, name).copy() if isinstance(getattr(self, name), (dict, set))
                    else getattr(self, name) for name in fields}
        try:
            return self._step(day, price)
        except Exception:
            for name, value in snapshot.items():
                setattr(self, name, value)
            raise

    def _step(self, day, price=None):
        day = pd.Timestamp(day)
        if day <= self.day:
            raise ValueError("Process each calendar date once, in order")
        marked = price is not None and math.isfinite(float(price)) and float(price) > 0
        for event in self.splits:
            when, known = pd.Timestamp(event["effective_date"]), pd.Timestamp(event["known_at"])
            if when <= self.day and known <= day and event["event_id"] not in self.applied_splits:
                raise ValueError("Late split disclosure requires inventory reconciliation")
            if self.day < when < day and known <= day:
                raise ValueError("Required split date skipped")
            if when != day or known > day or event["event_id"] in self.applied_splits:
                continue
            if event["ts_code"] != self.code or not event.get("primary_source_verified", False):
                raise ValueError("Split needs reviewed primary identity evidence")
            factor = float(event["new_units_per_old_unit"])
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError("Invalid split factor")
            self.quantity *= factor
            self.applied_splits.add(event["event_id"])
            if not marked:
                self.mark /= factor  # Provisional equivalent-unit mark, not an observed quote.
        self.history[day] = self.quantity
        for event in self.schedules.itertuples(index=False):
            if pd.Timestamp(event.known_at) > day:
                continue
            ex, record = pd.Timestamp(event.ex_date), pd.Timestamp(event.record_date)
            if self.day < ex < day:
                raise ValueError("Required ex-date skipped")
            if ex != day:
                continue
            if not event.eligible_exdate_research or event.event_id in self.claims or event.event_id in self.settled:
                raise ValueError("Unreviewed or repeated entitlement")
            if record not in self.history:
                raise ValueError("Record-date quantity is unknown")
            # Cash denomination across a split needs a separate reviewed contract.
            if any(record < pd.Timestamp(s["effective_date"]) <= ex and pd.Timestamp(s["known_at"]) <= day
                   for s in self.splits):
                raise ValueError("Split across record/ex-date requires entitlement denomination review")
            amount = float(event.cash_per_unit)
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("Invalid per-unit entitlement")
            self.claims[event.event_id] = self.history[record] * amount
        for receipt in self.receipts:
            if pd.Timestamp(receipt["receipt_available_at"]) > day or pd.Timestamp(receipt["actual_payment_date"]) > day:
                continue
            event_id = receipt["event_id"]
            if event_id in self.settled:
                continue
            if receipt["ts_code"] != self.code or not receipt.get("settlement_evidence_verified", False):
                raise ValueError("Cash receipt identity/evidence not verified")
            if event_id not in self.claims:
                raise ValueError("No corresponding receivable")
            amount = float(receipt["gross_cash_amount"])
            if not math.isclose(amount, self.claims[event_id], rel_tol=0, abs_tol=1e-8):
                raise ValueError("Receipt/entitlement amount mismatch")
            self.cash += self.claims.pop(event_id)
            self.settled.add(event_id)
        if marked:
            self.mark = float(price)
        self.day = day
        claim = sum(self.claims.values())
        value = self.quantity * self.mark + claim + self.cash
        return dict(date=day, quantity=self.quantity, cash=self.cash, receivable=claim,
            provisional_nav=value, observed_mark_nav=value if marked else None,
            missing_held_price=not marked, settled_events=len(self.settled),
            assumed_unit_position_scenario=True, position_and_cashflows_fully_certified=False)
