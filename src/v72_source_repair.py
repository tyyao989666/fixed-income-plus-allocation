"""Explicit source-identity quarantine, never a statistical return correction."""
import numpy as np
import pandas as pd


def quarantine_returns(frame, asset, through):
    """A reviewed bad source segment also invalidates its boundary return."""
    if asset not in frame or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Unique ordered calendar and exact asset identity required")
    result = frame.copy()
    result.loc[result.index <= pd.Timestamp(through), asset] = np.nan
    return result


def endpoint_reason(previous, current):
    before = np.isfinite(previous) and previous > 0
    after = np.isfinite(current) and current > 0
    if not before and not after:
        return "both_source_endpoints_missing"
    if not before:
        return "previous_source_endpoint_missing"
    if not after:
        return "current_source_endpoint_missing"
    return "both_source_endpoints_present_requires_other_evidence"
