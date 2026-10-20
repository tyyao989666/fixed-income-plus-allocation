"""Descriptive matched-date residual diagnostics, never used as trading signals."""
import numpy as np


def residual_alignment(prediction, residual, mask):
    x, y = np.asarray(prediction, float), np.asarray(residual, float)
    good = np.asarray(mask, bool) & np.isfinite(x) & np.isfinite(y)
    if x.shape != y.shape or x.shape != good.shape or x.ndim != 2:
        raise ValueError('Expected matching date/entity matrices')
    n = good.sum(1, keepdims=True)
    valid = n[:, 0] > 0
    if not valid.any():
        return dict(valid_dates=0, valid_cells=0, alignment=np.nan, raw_MSE_ratio=np.nan)
    w = good/np.maximum(n, 1)
    x, y = np.where(good, x, 0), np.where(good, y, 0)
    xy, xx, yy = (float(np.sum(w*v)) for v in [x*y, x*x, y*y])
    alignment = xy/np.sqrt(xx*yy) if xx > 1e-20 and yy > 1e-20 else np.nan
    ratio = float(np.sum(w*(x-y)**2))/yy if yy > 1e-20 else np.nan
    return dict(valid_dates=int(valid.sum()), valid_cells=int(good.sum()), alignment=alignment, raw_MSE_ratio=ratio)
