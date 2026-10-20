"""Announcement-aware dividend schedules, not evidence of settled cash."""
import hashlib

import numpy as np
import pandas as pd


def normalize_schedule(frame, code, cutoff):
    required = {"ts_code", "div_proc", "ann_date", "imp_anndate", "base_date",
                "record_date", "ex_date", "pay_date", "div_cash"}
    if not frame.empty and (required - set(frame) or not frame.ts_code.eq(code).all()):
        raise ValueError("Dividend source identity/schema mismatch")
    if frame.empty:
        return pd.DataFrame(), dict(source_rows=0, unique_events=0, eligible_events=0,
                                   source_empty_is_not_certified_zero_dividends=True)
    f = frame.copy()
    for name in ("ann_date", "imp_anndate", "base_date", "record_date", "ex_date", "pay_date"):
        f[name] = pd.to_datetime(f[name].astype(str), format="%Y%m%d", errors="coerce")
    f["known_at"] = f[["ann_date", "imp_anndate"]].max(axis=1)
    f["cash_per_unit"] = pd.to_numeric(f.div_cash, errors="coerce")
    f = f.loc[f.div_proc.eq("实施") & f.known_at.notna() & f.known_at.le(pd.Timestamp(cutoff))].copy()
    if f.empty:
        f["eligible_exdate_research"] = pd.Series(dtype=bool)
        f["settlement_certified"] = pd.Series(dtype=bool)
        return f, dict(source_rows=len(frame), unique_events=0, eligible_events=0,
                       rejected_or_unmatured_events=0, conflicting_events=0,
                       settlement_certified=False, reason="no_implemented_schedule_known_by_cutoff")
    keys = ["ts_code", "base_date", "record_date", "ex_date"]
    # Conflicting amounts or payment schedules require source review, not summing.
    conflicts = f.groupby(keys, dropna=False).agg(cash_values=("cash_per_unit", lambda x: x.nunique(dropna=False)),
                                                 pay_values=("pay_date", lambda x: x.nunique(dropna=False)))
    bad_keys = conflicts.index[(conflicts.cash_values > 1) | (conflicts.pay_values > 1)]
    f["conflicting_schedule"] = pd.MultiIndex.from_frame(f[keys]).isin(bad_keys)
    f = f.sort_values("known_at").drop_duplicates(keys, keep="last")
    f["eligible_exdate_research"] = (f.ex_date.notna() & f.record_date.notna() & f.pay_date.notna()
        & f.known_at.le(f.ex_date) & f.record_date.lt(f.ex_date) & f.pay_date.ge(f.ex_date)
        & f.ex_date.le(pd.Timestamp(cutoff)) & np.isfinite(f.cash_per_unit)
        & f.cash_per_unit.ge(0) & ~f.conflicting_schedule)
    f["payment_date_is_vendor_schedule_not_settlement_confirmation"] = True
    f["settlement_certified"] = False
    f["event_id"] = [hashlib.sha256("|".join(str(row[k]) for k in keys).encode()).hexdigest()[:20]
                      for _, row in f.iterrows()]
    return f, dict(source_rows=len(frame), unique_events=len(f), eligible_events=int(f.eligible_exdate_research.sum()),
        rejected_or_unmatured_events=int((~f.eligible_exdate_research).sum()),
        conflicting_events=int(f.conflicting_schedule.sum()), settlement_certified=False)
