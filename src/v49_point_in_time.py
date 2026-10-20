"""Announcement-aware observations and ex-ante convertible-bond buckets."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def next_session_after(days, sessions):
    sessions = pd.DatetimeIndex(sessions).normalize()
    dates = pd.DatetimeIndex(days).normalize()
    ix = sessions.searchsorted(dates, side="right")
    return pd.DatetimeIndex([sessions[i] if i < len(sessions) else pd.NaT for i in ix])


def cb_disclosure_events(frame, sessions, cutoff):
    required = {"ts_code", "publish_date", "end_date", "remain_size", "issue_size"}
    if required-set(frame):
        raise ValueError("CB release metadata missing")
    out = frame.copy()
    for col in ["publish_date", "end_date"]:
        out[col] = pd.to_datetime(out[col], errors="raise").dt.normalize()
    if out[["publish_date", "end_date"]].isna().any().any() or out.end_date.gt(out.publish_date).any():
        raise ValueError("CB statistical period cannot be ahead of announcement")
    out = out.loc[out.publish_date.le(pd.Timestamp(cutoff))].copy()
    for col in ["remain_size", "issue_size"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(out[col]).all() or out[col].lt(0).any():
            raise ValueError("Invalid disclosed balance")
    if out.remain_size.gt(out.issue_size*(1+1e-8)).any():
        raise ValueError("Outstanding balance exceeds disclosed issuance")
    out = out.drop_duplicates(["ts_code", "publish_date", "end_date", "remain_size", "issue_size"])
    if out.duplicated(["ts_code", "publish_date", "end_date"]).any():
        raise ValueError("Conflicting CB disclosure records")
    # Several reporting periods can be published together. Choose the latest
    # period actually disclosed, never a later publication of an older period.
    out = out.sort_values(["ts_code", "publish_date", "end_date"]).groupby(["ts_code", "publish_date"], as_index=False).tail(1).copy()
    out["available_at"] = next_session_after(out.publish_date, sessions)
    # Non-trading-day publications can collapse to one session as well.
    out = out.dropna(subset=["available_at"]).sort_values(["ts_code", "available_at", "publish_date", "end_date"])
    out = out.groupby(["ts_code", "available_at"], as_index=False).tail(1)
    newest_period = out.groupby("ts_code").end_date.cummax()
    out = out.loc[out.end_date.ge(newest_period)].copy()
    return out


def align_disclosed_balances(events, sessions, codes, max_age_days=120):
    dates = pd.DatetimeIndex(sessions)
    output = {name:pd.DataFrame(np.nan, index=dates, columns=codes) for name in ["balance", "issue_size", "age_days"]}
    for code in codes:
        part = events.loc[events.ts_code.eq(code)].sort_values("available_at")
        if part.empty:
            continue
        aligned = pd.merge_asof(pd.DataFrame({"date": dates}), part, left_on="date", right_on="available_at", direction="backward")
        age = (aligned.date-aligned.publish_date).dt.days
        eligible = age.between(0, max_age_days) & aligned.available_at.le(aligned.date)
        output["balance"][code] = aligned.remain_size.where(eligible).to_numpy()
        output["issue_size"][code] = aligned.issue_size.where(eligible).to_numpy()
        output["age_days"][code] = age.to_numpy()
    return output


def capped_weights(size, cap=.10):
    size = pd.to_numeric(size, errors="coerce")
    if not 0 < cap <= 1 or size.isna().any() or not np.isfinite(size).all() or size.le(0).any():
        raise ValueError("Capped weights need finite positive sizes")
    if len(size)*cap < 1-1e-12:
        raise ValueError("Concentration cap infeasible for number of members")
    weights = pd.Series(0., index=size.index)
    free, remaining = pd.Series(True, index=size.index), 1.
    for _ in range(len(size)+1):
        proposed = size[free]/size[free].sum()*remaining
        over = proposed > cap+1e-14
        if not over.any():
            weights.loc[proposed.index] = proposed
            break
        fixed = proposed.index[over]
        weights.loc[fixed] = cap
        free.loc[fixed] = False
        remaining = 1-weights[~free].sum()
        if not free.any():
            break
    if not np.isclose(weights.sum(), 1, atol=1e-10) or weights.max() > cap+1e-10:
        raise ValueError("Failed capped-simplex allocation")
    return weights


def cb_bucket_weights(signal_price, known_balance, known_issue_size, minimum_names=10, cap=.10):
    """No future-return/quote argument is allowed in constituent formation."""
    required = math.ceil(1/cap-1e-12)
    if minimum_names < required:
        raise ValueError("Minimum bucket size incompatible with concentration cap")
    p = signal_price.astype(float)
    balance, size = known_balance.reindex(p.index), known_issue_size.reindex(p.index)
    active = np.isfinite(p) & p.gt(0) & np.isfinite(balance) & balance.gt(0) & np.isfinite(size) & size.gt(0)
    median = size[active].median()
    styles = {"bond":active & p.le(110), "balanced":active & p.gt(110) & p.le(130), "equity":active & p.gt(130)}
    output = {}
    for style, scale, price in itertools.product(["bond", "balanced", "equity"], ["large", "small"], ["low", "high"]):
        coarse = styles[style] & (size.ge(median) if scale == "large" else size.lt(median))
        split = p[coarse].median()
        fine = coarse & (p.le(split) if price == "low" else p.gt(split))
        for level, candidate in enumerate([fine, coarse, styles[style], active]):
            if int(candidate.sum()) >= minimum_names:
                name = f"cb_{style}_{scale}_{price}"
                output[name] = dict(weights=capped_weights(balance[candidate], cap), fallback_level=level,
                                    membership_role="known_previous_close_and_last_published_balance")
                break
    return output


def realized_bucket_return(weights, realized_returns):
    """Missing held observations invalidate return, not the prior membership."""
    observed = realized_returns.reindex(weights.index)
    if not np.isfinite(observed).all() or observed.le(-1).any():
        return np.nan
    return float(weights @ observed)


def dividend_cash_per_unit(frame, dates, code):
    """Implemented, announced ex-date cash; identical records count once."""
    out = frame.loc[frame.ts_code.eq(code) & frame.div_proc.eq("实施")].copy()
    for name in ["ann_date", "imp_anndate", "ex_date", "base_date", "record_date", "pay_date"]:
        out[name] = pd.to_datetime(out[name].astype(str), format="%Y%m%d", errors="coerce")
    out["known_at"] = out[["ann_date", "imp_anndate"]].max(axis=1)
    out = out.loc[out.known_at.notna() & out.ex_date.notna() & out.known_at.le(out.ex_date)
                  & out.ex_date.le(pd.DatetimeIndex(dates).max())].copy()
    out["div_cash"] = pd.to_numeric(out.div_cash, errors="raise")
    if not np.isfinite(out.div_cash).all() or out.div_cash.lt(0).any():
        raise ValueError("Invalid per-unit cash dividend")
    keys = ["ts_code", "base_date", "record_date", "ex_date", "pay_date"]
    if (out.groupby(keys, dropna=False).div_cash.nunique() > 1).any():
        raise ValueError("Conflicting dividend events")
    out = out.drop_duplicates(keys)
    cash = out.groupby("ex_date").div_cash.sum().reindex(dates, fill_value=0.)
    return cash.astype(float)
