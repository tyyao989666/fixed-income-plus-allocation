"""Explicit retrospective research exclusions, not a historical default predictor."""
import pandas as pd


def apply_exclusions(eligibility, codes):
    if eligibility.index.has_duplicates or eligibility.columns.has_duplicates:
        raise ValueError("Unique dates/security identities required")
    if eligibility.isna().any().any() or not all(pd.api.types.is_bool_dtype(t) for t in eligibility.dtypes):
        raise ValueError("Eligibility must be an explicit boolean matrix")
    if len(set(codes)) != len(codes) or not set(codes).issubset(eligibility.columns):
        raise ValueError("Duplicate or unknown excluded security")
    result = eligibility.copy(deep=True)
    result.loc[:, list(codes)] = False
    return result
