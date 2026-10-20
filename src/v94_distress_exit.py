"""Pure prior-close risk overlay. Unknown/frozen prices do not authorize sales."""
import re

import numpy as np
import pandas as pd

TRIGGERS = {"court_accepts_issuer_reorganization", "exchange_decides_security_delisting"}


def visible_exit_mask(codes, signal_date, notices):
    if len(set(codes)) != len(codes):
        raise ValueError("Unique security identities required")
    day = pd.Timestamp(signal_date)
    if pd.isna(day):
        raise ValueError("Signal date required")
    out = np.zeros(len(codes), dtype=bool)
    for row in notices:
        if row["ts_code"] not in codes or row["trigger_type"] not in TRIGGERS:
            raise ValueError("Unexpected security or trigger type")
        known = pd.Timestamp(row["known_at"])
        if pd.isna(known) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Dated primary evidence hash required")
        if known <= day:
            out[codes.index(row["ts_code"])] = True
    return out


def adjust_targets(target, buy_allowed, codes, signal_date, notices):
    target = np.asarray(target, dtype=float)
    allowed = np.asarray(buy_allowed, dtype=bool)
    if target.ndim != 2 or target.shape[1] != len(codes) or allowed.shape != (len(codes),):
        raise ValueError("Target/permission coordinates differ")
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid long-only targets")
    totals = target.sum(1)
    if not (np.isclose(totals, 0) | np.isclose(totals, 1)).all():
        raise ValueError("Targets must be simplexes or unavailable")
    blocked = visible_exit_mask(codes, signal_date, notices)
    revised = target.copy()
    affected_rows = (target[:, blocked] > 0).any(axis=1)
    revised[:, blocked] = 0
    remaining = revised.sum(1)
    np.divide(revised, remaining[:, None], out=revised,
              where=(remaining[:, None] > 0) & affected_rows[:, None])
    # If the original target has no surviving instrument, retain an unavailable
    # target. The execution book must not invent a cash investment or liquidation.
    return revised, allowed & ~blocked, blocked
