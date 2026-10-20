"""Research fund portfolio ledger: reuse fills, add entitlements and T+1 provenance."""
import math

import numpy as np
import pandas as pd

from src.v51_cb_policy import Inventory


class FundPortfolio:
    """Price-only fund sources only. Never pass exdate-total-return NAV here."""
    def __init__(self, codes, opening_date, cash, schedules, splits=None, receipts=None, cost_rate=.0005, sessions=None):
        index = pd.Index(codes)
        if not index.is_unique or not np.isfinite(cash) or cash <= 0 or not 0 <= cost_rate < 1:
            raise ValueError("Unique fund codes, finite positive cash and valid cost required")
        self.book = Inventory(pd.Series(0., index=index), pd.Series(np.nan, index=index), float(cash))
        self.day, self.cost_rate = pd.Timestamp(opening_date), float(cost_rate)
        self.schedules = schedules.copy()
        if not schedules.empty and (not schedules.ts_code.isin(index).all() or schedules.event_id.duplicated().any()):
            raise ValueError("Distribution identities must belong to the exact fund universe")
        self.splits, self.receipts = list(splits or []), list(receipts or [])
        self.history = {self.day: self.book.quantity.copy()}
        self.claims, self.applied_splits, self.settled = {}, set(), set()
        self.sessions = pd.DatetimeIndex(sessions) if sessions is not None else None
        if self.sessions is not None and (self.sessions.has_duplicates or not self.sessions.is_monotonic_increasing):
            raise ValueError("Explicit domestic sessions must be ordered and unique")

    def step(self, day, prices, volume, *, target=None, signal_date=None, signal_buy_allowed=None):
        snapshot = self.__dict__.copy()
        snapshot["book"] = Inventory(self.book.quantity.copy(), self.book.last_mark.copy(), self.book.cash)
        snapshot["history"], snapshot["claims"] = self.history.copy(), self.claims.copy()
        snapshot["applied_splits"], snapshot["settled"] = self.applied_splits.copy(), self.settled.copy()
        try:
            return self._step(day, prices, volume, target, signal_date, signal_buy_allowed)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise

    def _step(self, day, prices, volume, target, signal_date, signal_buy_allowed):
        day = pd.Timestamp(day)
        if day <= self.day or not prices.index.equals(self.book.quantity.index) or not volume.index.equals(prices.index):
            raise ValueError("Ordered dates and exact price/volume coordinates required")
        if target is not None:
            if (signal_date is None or pd.Timestamp(signal_date) > self.day or pd.Timestamp(signal_date) >= day
                    or signal_buy_allowed is None or not signal_buy_allowed.index.equals(prices.index)):
                raise ValueError("Only a prior processed close may supply target and buy eligibility")
            if self.sessions is None or day not in self.sessions:
                raise ValueError("Explicit domestic calendar required for target execution")
            position = self.sessions.get_loc(day)
            if position == 0 or self.sessions[position - 1] != pd.Timestamp(signal_date):
                raise ValueError("Target must execute on exactly the next domestic session")
        for split in self.splits:
            when, known = pd.Timestamp(split["effective_date"]), pd.Timestamp(split["known_at"])
            if known > day:
                continue
            if when <= self.day and split["event_id"] not in self.applied_splits:
                raise ValueError("Late split requires inventory reconciliation")
            if self.day < when < day:
                raise ValueError("Required split date skipped")
            if when == day and split["event_id"] not in self.applied_splits:
                code, factor = split["ts_code"], float(split["new_units_per_old_unit"])
                if code not in prices.index or not split.get("primary_source_verified") or not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Split identity/factor/source not reviewed")
                self.book.quantity.loc[code] *= factor
                self.book.last_mark.loc[code] /= factor
                self.applied_splits.add(split["event_id"])
        for event in self.schedules.itertuples(index=False):
            if pd.Timestamp(event.known_at) > day:
                continue
            ex, record = pd.Timestamp(event.ex_date), pd.Timestamp(event.record_date)
            if self.day < ex < day:
                raise ValueError("Required ex-date skipped")
            if ex != day:
                continue
            if not event.eligible_exdate_research or event.event_id in self.claims or event.event_id in self.settled:
                raise ValueError("Invalid or duplicate dividend event")
            if record not in self.history:
                raise ValueError("Missing record-date inventory; do not infer entitlement")
            if any(record < pd.Timestamp(s["effective_date"]) <= ex and pd.Timestamp(s["known_at"]) <= day
                   and s["ts_code"] == event.ts_code for s in self.splits):
                raise ValueError("Entitlement denomination across split not certified")
            cash = float(event.cash_per_unit)
            if not math.isfinite(cash) or cash < 0:
                raise ValueError("Invalid per-unit cash distribution")
            self.claims[event.event_id] = float(self.history[record].loc[event.ts_code] * cash)
        for receipt in self.receipts:
            if pd.Timestamp(receipt["receipt_available_at"]) > day or pd.Timestamp(receipt["actual_payment_date"]) > day:
                continue
            event_id = receipt["event_id"]
            if event_id in self.settled:
                continue
            if (event_id not in self.claims or not receipt.get("settlement_evidence_verified")
                    or receipt["ts_code"] not in prices.index
                    or not math.isclose(float(receipt["gross_cash_amount"]), self.claims[event_id], abs_tol=1e-8, rel_tol=0)):
                raise ValueError("Receipt evidence/amount does not match the booked claim")
            expected = self.schedules.loc[self.schedules.event_id.eq(event_id), "ts_code"]
            if len(expected) != 1 or expected.iloc[0] != receipt["ts_code"]:
                raise ValueError("Receipt security differs from registered entitlement")
            self.book.cash += self.claims.pop(event_id)
            self.settled.add(event_id)
        observed = np.isfinite(prices) & prices.gt(0)
        before_marks = prices.where(observed, self.book.last_mark)
        before = float((self.book.quantity * before_marks.fillna(0)).sum() + self.book.cash + sum(self.claims.values()))
        cost = turnover_notional = 0.
        fills = pd.Series(0., index=prices.index)
        executable = observed & np.isfinite(volume) & volume.gt(0)
        if target is not None:
            # Allocate only currently investable capital; claims are never buying power.
            result = self.book.close_step(target, prices, executable, prices,
                                          signal_buy_allowed, cost_rate=self.cost_rate)
            fills, cost = result["fills"], float(result["cost"])
            turnover_notional = float((fills.abs() * prices.fillna(0)).sum())
        else:
            self.book.last_mark = before_marks
        marked = bool((observed | self.book.quantity.eq(0)).all())
        nav = float((self.book.quantity * self.book.last_mark.fillna(0)).sum() + self.book.cash + sum(self.claims.values()))
        if not np.isclose(nav, before - cost, atol=1e-7, rtol=1e-10) or self.book.cash < -1e-8:
            raise ValueError("Self-financing conservation failed or negative spendable cash")
        self.history[day] = self.book.quantity.copy()  # Post-trade record-date ownership.
        self.day = day
        return dict(date=day, provisional_nav=nav, observed_mark_nav=nav if marked else np.nan,
            spendable_cash=float(self.book.cash), receivable=float(sum(self.claims.values())),
            cost=cost, gross_traded_notional=turnover_notional,
            gross_turnover_nav=turnover_notional / before if before > 0 else np.nan,
            frozen_market_value=float((self.book.quantity.where(~executable, 0) * self.book.last_mark.fillna(0)).sum()),
            fills=fills.copy(), quantities=self.book.quantity.copy(),
            settled_event_count=len(self.settled), signal_date=signal_date,
            fractional_unit_research_execution=True, cashflow_and_execution_fully_certified=False)
