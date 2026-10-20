"""Group-centered rank-to-price-return bridge, independent of V61 training."""
import numpy as np


def centered_scores(scores, eligible, groups):
    scores = np.asarray(scores, float)
    eligible = np.asarray(eligible, bool)
    if scores.shape != eligible.shape or scores.ndim != 3:
        raise ValueError('Expected matching date/horizon/entity arrays')
    if len(groups) != scores.shape[-1]:
        raise ValueError('Group shape mismatch')
    if not np.isfinite(scores[eligible]).all():
        raise ValueError('Nonfinite eligible score')
    result = np.zeros_like(scores)
    for group in np.unique(groups):
        members = groups == group
        x, m = scores[:, :, members], eligible[:, :, members]
        n = m.sum(-1, keepdims=True)
        mean = np.divide(np.where(m, x, 0).sum(-1, keepdims=True), n,
                         out=np.zeros_like(n, dtype=float), where=n > 0)
        result[:, :, members] = np.where(m & (n >= 2), x - mean, 0)
    return result


def fit_centered_slopes(centered, truth, label_mask, prior, groups,
                        ridge=0.1, minimum_dates=100):
    if ridge < 0 or minimum_dates < 1:
        raise ValueError('Invalid calibration controls')
    slopes = np.zeros(prior.shape, dtype=float)
    rows = []
    for h in range(centered.shape[1]):
        for group in np.unique(groups):
            members = groups == group
            x = centered[:, h, members]
            y = truth[:, h, members] - prior[h, members]
            good = label_mask[:, h, members] & np.isfinite(x) & np.isfinite(y)
            n = good.sum(-1, keepdims=True)
            paired_dates = (n[:, 0] >= 2)
            good &= paired_dates[:, None]
            weights = good / np.maximum(n, 1)
            numerator = float(np.sum(weights * np.where(good, x * y, 0)))
            denominator = float((1 + ridge) * np.sum(weights * np.where(good, x*x, 0)))
            reason = 'positive'
            beta = 0.
            if paired_dates.sum() < minimum_dates:
                reason = 'insufficient_paired_dates'
            elif denominator <= 1e-12:
                reason = 'no_score_dispersion'
            elif numerator <= 0:
                reason = 'nonpositive_calibration_relation'
            else:
                beta = numerator / denominator
            slopes[h, members] = beta
            rows.append(dict(horizon_index=h, group_index=int(group), slope=beta,
                             paired_dates=int(paired_dates.sum()), reason=reason))
    return slopes, rows


def comparison_mask(eligible_labels, prediction, baseline=None):
    result = np.asarray(eligible_labels, bool) & np.isfinite(prediction)
    if baseline is not None:
        result &= np.isfinite(baseline)
    return result
