"""Read-only CB source contracts; historical observations are not PIT releases."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from src.v49_point_in_time import next_session_after


def panel(frame, start="2018-01-01", cutoff="2026-10-14"):
    frame = frame.copy()
    key = next((x for x in ("date", "trade_date", "日期") if x in frame), None)
    if key is not None:
        frame = frame.set_index(key)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="raise")
    frame.index = frame.index.normalize()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.index.hasnans or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Duplicate or invalid source panel keys")
    frame = frame.sort_index().loc[start:cutoff].apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Infinite source observation")
    frame.index.name = "date"
    return frame


def historical_universe(basic, columns, cutoff="2026-10-14"):
    """Do not select by today's outstanding balance, rating or survivor list.

    Exchange code/type screening is provisional public-issuance identification;
    a historical issuance-method registry is still required for certification.
    """
    required = {"ts_code", "cb_type", "bond_short_name", "list_date", "issue_size"}
    if required-set(basic) or basic.ts_code.duplicated().any():
        raise ValueError("Ambiguous historical CB master")
    out = basic.copy().set_index("ts_code").reindex(columns)
    out["list_date_parsed"] = pd.to_datetime(out.list_date.astype(str), errors="coerce")
    out["issue_size"] = pd.to_numeric(out.issue_size, errors="coerce")
    standard = pd.Series([bool(re.fullmatch(r"(?:110|111|113|118|123|125|127|128)\d{3}\.(?:SH|SZ)", x))
                          for x in out.index], index=out.index)
    standard &= out.cb_type.eq("CB") & ~out.bond_short_name.fillna("").str.contains(
        "定向|定转|定[0-9一二三四五六七八九十]|可交换|分离")
    standard &= out.list_date_parsed.notna() & out.list_date_parsed.le(pd.Timestamp(cutoff))
    standard &= np.isfinite(out.issue_size) & out.issue_size.gt(0)
    out["included_research_universe"] = standard
    out["public_issuance_history_verified"] = False
    return out


def observed_returns(close, preclose):
    preclose = preclose.reindex(index=close.index, columns=close.columns)
    valid = close.gt(0) & preclose.gt(0)
    # No winsorization or filtering membership by the realized move.
    return (close/preclose-1).where(valid)


def rebalance_turnover(previous, previous_return, target):
    """Half-L1 one-way turnover versus the marked-to-market previous book."""
    if previous is None:
        return 1.0
    realized = previous_return.reindex(previous.index)
    if not np.isfinite(realized).all() or realized.le(-1).any():
        return np.nan
    marked = previous*(1+realized)
    if marked.sum() <= 0:
        return np.nan
    marked /= marked.sum()
    names = marked.index.union(target.index)
    return float((marked.reindex(names, fill_value=0)-target.reindex(names, fill_value=0)).abs().sum()/2)


def strict_nav(returns, base=100.):
    """No synthetic zero return or implicit bridge across unresolved marks."""
    return (1+returns).cumprod(skipna=False)*base


def known_exit_exclusions(calls, master, dates, maturity_buffer_days=7):
    """Quarantine announced implemented calls; never infer exits from last quote.

    Announcement-day time is unknown: effective on the next trading session.
    Contract maturity is fixed issue information, not a future delisting date.
    A seven-calendar-day maturity buffer is a research rule, not optimized.
    """
    if maturity_buffer_days < 0:
        raise ValueError("Invalid maturity buffer")
    dates = pd.DatetimeIndex(dates)
    mask = pd.DataFrame(False,index=dates,columns=master.index)
    maturity = pd.to_datetime(master.maturity_date.astype(str),errors="coerce")
    for code in mask:
        if pd.notna(maturity[code]):
            mask[code] = dates >= maturity[code]-pd.Timedelta(days=maturity_buffer_days)
    required = {"ts_code","call_type","is_call","ann_date"}
    if required-set(calls):
        raise ValueError("Dated call metadata required")
    events = calls.loc[calls.call_type.eq("强赎") & calls.is_call.isin(["公告实施强赎","公告不强赎"])].copy()
    # Tushare YYYYMMDD and already-normalized date objects are both supported.
    events["ann_date"] = pd.to_datetime(events.ann_date.astype(str),errors="raise").dt.normalize()
    if events.ann_date.isna().any():
        raise ValueError("Call announcement date missing")
    events = events.loc[events.ann_date.le(dates.max()) & events.ts_code.isin(mask.columns)]
    events["quarantine"] = events.is_call.eq("公告实施强赎")
    # Same-day conflicting vendor statuses are conservatively quarantined.
    events = events.groupby(["ts_code","ann_date"],as_index=False).quarantine.max()
    events["available_at"] = next_session_after(events.ann_date,dates)
    events = events.dropna(subset=["available_at"]).sort_values(["ts_code","available_at","ann_date"])
    events = events.drop_duplicates(["ts_code","available_at"],keep="last")
    for code,part in events.groupby("ts_code"):
        aligned = pd.merge_asof(pd.DataFrame({"date":dates}),part,
            left_on="date",right_on="available_at",direction="backward")
        mask[code] |= aligned.quarantine.eq(True).to_numpy()
    return mask,events
