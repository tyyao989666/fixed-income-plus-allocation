"""Check issuer aggregate against per-bond terms without overriding contradictions."""
from decimal import Decimal


def reconcile_redemption_total(bonds, per_bond, aggregate, decimals):
    quantity, unit, total = map(lambda x: Decimal(str(x)), (bonds, per_bond, aggregate))
    if (not all(x.is_finite() for x in [quantity, unit, total]) or quantity <= 0
            or quantity != quantity.to_integral_value() or unit <= 0 or total <= 0
            or not isinstance(decimals, int) or not 0 <= decimals <= 6):
        raise ValueError("Invalid aggregate payment evidence")
    expected = quantity * unit
    rounding_bound = quantity * Decimal(10) ** (-decimals) / 2 + Decimal("0.01")
    discrepancy = total - expected
    return dict(expected_gross_yuan=float(expected), reported_gross_yuan=float(total),
                discrepancy_yuan=float(discrepancy), maximum_rounding_difference_yuan=float(rounding_bound),
                aggregate_reconciled=bool(abs(discrepancy) <= rounding_bound))
