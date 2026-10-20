"""Non-posting creditor-scope diagnostics; these are not recovery valuations."""
from decimal import Decimal, InvalidOperation


def amount(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid amount") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Amount must be finite and nonnegative")
    return result


def creditor_tiers(claims, cash_cap, stock_cap=None, shares_per_100=None):
    """One cap per legally bound creditor, not per basket or custody account.

    Only supports cash-first then stock then unpriced residual rights. A caller
    must supply confirmed aggregate claims, not infer them from bond face value.
    No allocation of a creditor's cash to individual baskets is assumed.
    """
    cap = amount(cash_cap)
    ceiling = amount(stock_cap) if stock_cap is not None else None
    ratio = amount(shares_per_100) if shares_per_100 is not None else None
    if (ceiling is None) != (ratio is None) or (ceiling is not None and ceiling < cap):
        raise ValueError("Stock tier requires a ceiling and ratio, above cash cap")
    totals = {}
    seen = set()
    plan_ids = set()
    for row in claims:
        if not row.get("creditor_id") or not row.get("claim_id") or not row.get("plan_id"):
            raise ValueError("Plan, legal creditor and unique accepted-claim IDs required")
        plan_ids.add(row["plan_id"])
        if len(plan_ids) > 1:
            raise ValueError("Do not combine independent restructuring plans")
        if row["claim_id"] in seen:
            raise ValueError("Duplicate claim would consume cap twice")
        if row.get("scope_verified") is not True or row.get("aggregate_complete") is not True:
            raise ValueError("Unverified or incomplete creditor scope")
        seen.add(row["claim_id"])
        key = row["creditor_id"]
        totals[key] = totals.get(key, Decimal(0)) + amount(row["accepted_claim_yuan"])
    result = []
    for creditor, total in sorted(totals.items()):
        cash = min(total, cap)
        stock_basis = min(max(total - cap, Decimal(0)), ceiling - cap) if ceiling is not None else Decimal(0)
        result.append(dict(creditor_id=creditor, accepted_claim_yuan=total,
                           contractual_cash_tier_yuan=cash,
                           stock_claim_basis_yuan=stock_basis,
                           indicative_share_units=stock_basis / 100 * ratio if ratio is not None else None,
                           residual_unpriced_claim_basis_yuan=total - cash - stock_basis,
                           cash_received_yuan=None, recovery_value_yuan=None,
                           posting_allowed=False))
    return result
