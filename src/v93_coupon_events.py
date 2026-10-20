"""Attach individually reviewed coupon schedules, without assuming payment."""
import pandas as pd

from src.v82_causal_cb_events import validate_causal_events


def augment_coupon_events(existing, coupons, codes):
    extra = pd.DataFrame(coupons)
    if extra.empty or not extra.kind.eq("coupon").all():
        raise ValueError("Only explicit coupon schedules supported")
    if extra.confirmed_at.notna().any():
        raise ValueError("This schedule-only batch must not fabricate confirmation")
    combined = validate_causal_events(pd.concat([existing, extra], ignore_index=True), codes)
    for row in extra.itertuples(index=False):
        terminal = combined[(combined.ts_code == row.ts_code) & (combined.kind == "terminal")]
        if (terminal.ex_date <= pd.Timestamp(row.ex_date)).any():
            raise ValueError("Coupon overlaps retired security/terminal final interest")
    return combined
