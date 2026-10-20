"""Research eligibility, masked labels and quantity-preserving inventory primitives."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def buy_eligibility(close, volume, balance, issue_size, excluded):
    """Inputs are a SINGLE signal-close snapshot, never execution-day evidence."""
    fields = [volume, balance, issue_size, excluded]
    if any(not x.index.equals(close.index) for x in fields):
        raise ValueError("Snapshot identities must align exactly")
    reasons = pd.DataFrame({
        "missing_signal_price": ~np.isfinite(close) | close.le(0),
        "no_confirmed_signal_trading": ~np.isfinite(volume) | volume.le(0),
        "missing_positive_disclosed_balance": ~np.isfinite(balance) | balance.le(0),
        "missing_positive_issue_size": ~np.isfinite(issue_size) | issue_size.le(0),
        "known_exit_or_unknown_exit_status": excluded.astype("boolean").fillna(True).astype(bool),
    })
    return ~reasons.any(axis=1), reasons


def masked_forward_labels(returns, horizons=(5, 20, 60)):
    """Keep every date/asset; false-mask zeros are storage placeholders only."""
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Calendar must be unique and increasing")
    r = returns.to_numpy(dtype=float)
    valid = np.isfinite(r) & (r > -1)
    values = np.zeros((len(r), len(horizons), r.shape[1]), dtype=np.float32)
    masks = np.zeros_like(values, dtype=bool)
    for j, horizon in enumerate(horizons):
        if horizon <= 0:
            raise ValueError("Positive horizon required")
        for t in range(len(r) - horizon):
            mask = valid[t + 1:t + horizon + 1].all(axis=0)
            growth = np.prod(1 + np.where(valid[t + 1:t + horizon + 1],
                                        r[t + 1:t + horizon + 1], 0), axis=0) - 1
            masks[t, j] = mask
            values[t, j] = np.where(mask, growth, 0)
    return values, masks


@dataclass
class Inventory:
    """Close execution ledger. Cash is settlement bookkeeping, not an asset sleeve.

    Marks never authorize execution. Unknown valuation keeps the quantity and a
    provisional last mark, but invalidates certified NAV until resolved. External
    corporate-action adjudication must separately certify terminal cashflows.
    """
    quantity: pd.Series
    last_mark: pd.Series
    cash: float

    def close_step(self, target, execution_price, executable, valuation_price,
                   signal_buy_allowed, cost_rate=.0005):
        index = self.quantity.index
        for item in (target, execution_price, executable, valuation_price, signal_buy_allowed):
            if not item.index.equals(index):
                raise ValueError("Ledger identities must align")
        if (not np.isfinite(target).all() or target.lt(0).any()
                or not np.isclose(target.sum(), 1) or not 0 <= cost_rate < 1
                or self.cash < -1e-10 or not np.isfinite(self.cash)
                or not np.isfinite(self.quantity).all() or self.quantity.lt(0).any()):
            raise ValueError("Invalid long-only ledger/target/cost")
        observed = np.isfinite(valuation_price) & valuation_price.gt(0)
        # An executable transaction is also a valid closing mark.
        tradable = executable.fillna(False).astype(bool) & np.isfinite(execution_price) & execution_price.gt(0)
        marks = valuation_price.where(observed, self.last_mark)
        marks = marks.where(~tradable, execution_price)
        certified = bool((observed | tradable | self.quantity.eq(0)).all())
        if (self.quantity.gt(0) & (~np.isfinite(marks) | marks.le(0))).any():
            raise ValueError("Held position has no provisional mark")
        nav_before = float((self.quantity * marks.fillna(0)).sum() + self.cash)
        if nav_before <= 0:
            raise ValueError("Non-positive provisional NAV")
        desired = target * nav_before / marks.where(marks.gt(0))
        requested = desired.sub(self.quantity).fillna(0)
        fills = pd.Series(0., index=index)
        sells = requested.clip(upper=0).where(tradable, 0)
        proceeds = float((-sells * execution_price.fillna(0)).sum())
        self.cash += proceeds * (1 - cost_rate)
        self.quantity += sells
        fills += sells
        buys = requested.clip(lower=0).where(tradable & signal_buy_allowed.fillna(False).astype(bool), 0)
        needed = float((buys * execution_price.fillna(0)).sum()) * (1 + cost_rate)
        scale = min(1., max(0., self.cash) / needed) if needed > 0 else 0.
        buys *= scale
        self.cash -= float((buys * execution_price.fillna(0)).sum()) * (1 + cost_rate)
        self.quantity += buys
        fills += buys
        self.last_mark = marks.copy()
        cost = float((fills.abs() * execution_price.fillna(0)).sum()) * cost_rate
        provisional = float((self.quantity * marks.fillna(0)).sum() + self.cash)
        return dict(fills=fills, frozen_quantity=self.quantity.where(~tradable, 0).copy(),
                    cost=cost, provisional_nav=provisional,
                    certified_nav=provisional if certified else np.nan,
                    valuation_complete=certified, settlement_cash=self.cash)
