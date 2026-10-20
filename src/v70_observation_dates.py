"""Map monthly labels to the actual final session without inventing weekend quotes."""
import pandas as pd


def month_end_session(label, sessions):
    label=pd.Timestamp(label)
    dates=pd.DatetimeIndex(sessions)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Ordered unique sessions required')
    matches=dates[(dates.to_period('M')==label.to_period('M')) & (dates<=label)]
    if not len(matches):
        raise ValueError('No actual observation in the labelled month')
    return matches[-1]
