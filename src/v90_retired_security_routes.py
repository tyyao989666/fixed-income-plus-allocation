"""Point-in-time data-query routes, not permissions to execute OTC trades."""
import re

import pandas as pd


def route_at_signal(old_code, signal_date, routes):
    day = pd.Timestamp(signal_date)
    if pd.isna(day):
        raise ValueError("Signal date required")
    visible = []
    for row in routes:
        if row["old_code"] != old_code:
            continue
        known, start = pd.Timestamp(row["known_at"]), pd.Timestamp(row["transfer_start"])
        if pd.isna(known) or pd.isna(start) or not re.fullmatch(r"\d{6}", row["new_code"]):
            raise ValueError("Invalid dated security route")
        if not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
            raise ValueError("Evidence hash required")
        if known <= day and start <= day:
            visible.append(row)
    if len(visible) > 1:
        raise ValueError("Ambiguous route revisions; explicit validity intervals required")
    if not visible:
        return dict(query_code=old_code, venue="ORIGINAL", routed=False,
                    execution_authorized=False, rights_confirmed=False)
    row = visible[0]
    return dict(query_code=row["new_code"], venue="NEEQ_DELISTED", routed=True,
                execution_authorized=False, rights_confirmed=False,
                scheduled_weekdays=row["scheduled_weekdays"],
                source_sha256=row["source_sha256"])


def conditional_cash_reference(quantity, unit_cash, asof, confirmation_known_at, *, assumptions_explicit):
    """Isolated gross-cash scenario, never a receipt or reinvestable book balance."""
    from src.v89_distressed_claims import amount
    qty, unit = amount(quantity), amount(unit_cash)
    day, known = pd.Timestamp(asof), pd.Timestamp(confirmation_known_at)
    if pd.isna(day) or pd.isna(known):
        raise ValueError("Scenario requires a confirmation knowledge date")
    if assumptions_explicit is not True:
        raise ValueError("Creditor scope/claim acceptance must be explicitly assumed")
    return dict(scenario_gross_cash_reference_yuan=float(qty * unit) if day >= known else None,
                actual_cash_received_yuan=None, certified=False, posting_allowed=False)
