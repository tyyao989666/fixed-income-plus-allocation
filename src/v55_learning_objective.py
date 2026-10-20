"""Causal temporal opportunity targets, invertible to raw relative-return views."""
import numpy as np
import pandas as pd


def temporal_opportunity_targets(raw_target, raw_mask, daily_returns, horizons=(5,20,60)):
    """No future label may contribute to a signal-date location estimate.

    z(t,h) = raw_excess(t,h)/(past_vol(t)*sqrt(h)) - location(t,h).
    Location uses only labels whose end is strictly BEFORE t, with a 252-date
    rolling mean (minimum126), shrunk by n/(n+252). The risk scale uses t-60..t-1,
    minimum40 observed daily returns, annual vol floor1%. These are fixed design
    choices, not selected by OOS performance. Missing inputs invalidate a cell.
    """
    y, mask = np.asarray(raw_target,dtype=float), np.asarray(raw_mask,dtype=bool)
    daily = pd.DataFrame(np.asarray(daily_returns,dtype=float)).replace([np.inf,-np.inf],np.nan)
    if y.ndim != 3 or mask.shape != y.shape or daily.shape != (y.shape[0],y.shape[2]):
        raise ValueError("Target/return shapes disagree")
    if len(horizons) != y.shape[1]:
        raise ValueError("Horizon count disagrees")
    vol = daily.shift(1).rolling(60,min_periods=40).std(ddof=0).clip(lower=.01/np.sqrt(252))
    output = np.zeros_like(y,dtype=np.float32)
    valid = np.zeros_like(mask)
    scales = np.zeros_like(y,dtype=np.float32)
    offsets = np.zeros_like(y,dtype=np.float32)
    source_last_index = np.full((len(y),len(horizons)),-1,dtype=np.int64)
    for j,h in enumerate(horizons):
        scale = vol * np.sqrt(h)
        score = pd.DataFrame(np.where(mask[:,j],y[:,j],np.nan)) / scale
        # Row s matures at s+h; shifting h+1 ensures s+h <= t-1.
        matured = score.shift(h+1)
        rolling = matured.rolling(252,min_periods=126)
        count = rolling.count()
        location = rolling.mean() * count/(count+252)
        available = np.isfinite(scale.to_numpy()) & np.isfinite(location.to_numpy())
        good = mask[:,j] & np.isfinite(score.to_numpy()) & available
        valid[:,j] = good
        output[:,j] = np.where(good,score.to_numpy()-location.to_numpy(),0)
        # Parameters remain available for current unlabeled forecast dates too.
        scales[:,j] = np.where(available,scale.to_numpy(),0)
        offsets[:,j] = np.where(available,location.to_numpy(),0)
        source_last_index[:,j] = np.maximum(np.arange(len(y))-h-1,-1)
    return dict(target=output,mask=valid,scale=scales,offset=offsets,
                latest_location_source_index=source_last_index)


def decode_opportunity(score, scale, offset):
    values, scales, offsets = np.broadcast_arrays(score,scale,offset)
    return np.where(np.isfinite(scales)&(scales>0)&np.isfinite(offsets),
                    (values+offsets)*scales,np.nan)
