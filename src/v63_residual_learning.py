"""Maturity-aware historical control and a supervised residual objective."""
import numpy as np
import torch
from src.v61_state_momentum import rank_loss


def historical_control(dates, label_end, truth, mask, groups, sources, queries,
                       history_sessions=756, minimum_group_dates=60,
                       minimum_entity_labels=60, shrinkage=126):
    dates, label_end = np.asarray(dates), np.asarray(label_end)
    truth, mask = np.asarray(truth), np.asarray(mask, bool)
    sources, queries = np.asarray(sources), np.asarray(queries)
    if truth.shape != mask.shape or label_end.shape != truth.shape[:2]:
        raise ValueError('Invalid target/maturity shapes')
    if np.any(np.diff(dates) <= 0) or np.any(sources < 0) or np.any(sources >= len(dates)):
        raise ValueError('Invalid chronological inputs')
    if len(groups) != truth.shape[-1] or len(np.unique(sources)) != len(sources):
        raise ValueError('Invalid groups or duplicate source rows')
    result = np.full((len(queries), *truth.shape[1:]), np.nan, dtype=np.float32)
    counts = np.zeros(result.shape, np.int32)
    latest = np.zeros((len(queries), truth.shape[1]), np.int64)
    for j, query in enumerate(queries):
        for h in range(truth.shape[1]):
            # Strictly earlier maturity: never consume a label finishing today.
            selected = sources[(sources < query) & (sources >= query-history_sessions)
                & (label_end[sources, h] > 0) & (label_end[sources, h] < dates[query])]
            if not len(selected):
                continue
            y = truth[selected, h]
            good = mask[selected, h] & np.isfinite(y)
            count = good.sum(0)
            counts[j, h] = count
            used = good.any(1)
            if used.any():
                latest[j, h] = label_end[selected[used], h].max()
            for group in np.unique(groups):
                members = groups == group
                m, v = good[:, members], y[:, members]
                n = m.sum(1)
                valid = n > 0
                if valid.sum() < minimum_group_dates:
                    continue
                daily = np.where(m, v, 0).sum(1) / np.maximum(n, 1)
                center = daily[valid].mean()
                c = count[members]
                own = np.divide(np.where(m, v, 0).sum(0), c,
                    out=np.full(len(c), center), where=c > 0)
                weight = np.where(c >= minimum_entity_labels, c/(c+shrinkage), 0)
                result[j, h, members] = weight*own + (1-weight)*center
    if np.any(latest >= dates[queries, None]):
        raise AssertionError('Historical control consumed an immature label')
    return result, counts, latest


def fit_residual_scale(residual, mask, groups):
    scale = np.full(residual.shape[1:], np.nan, np.float32)
    for h in range(residual.shape[1]):
        for group in np.unique(groups):
            members = groups == group
            good = mask[:, h, members] & np.isfinite(residual[:, h, members])
            values = residual[:, h, members][good]
            if len(values) >= 2:
                scale[h, members] = max(float(values.std()), 1e-4)
    return scale


def bounded_increment(pred, bound=1.):
    return {t: bound*torch.tanh(v/bound) for t, v in pred.items()}


def residual_loss(pred, targets, masks, groups, rank_weight=.05, penalty=.01):
    total = pred['asset'].sum()*0
    for task in pred:
        for h, weight in enumerate([.15, .50, .35]):
            terms = []
            for group in groups[task].unique():
                members = groups[task] == group
                m = masks[task][:, h, members].bool()
                n = m.sum(1)
                if not (n > 0).any():
                    continue
                p = pred[task][:, h, members]
                y = torch.where(m, targets[task][:, h, members], 0)
                err = (p-y).square() + penalty*p.square()
                terms.append(((err*m).sum(1)/n.clamp_min(1))[n > 0].mean())
            if terms:
                total = total+.5*weight*torch.stack(terms).mean()
    if rank_weight:
        total = total+rank_weight*rank_loss(pred, targets, masks, groups, penalty=0.)
    return total


def fit_gain(prediction, residual, mask, groups, ridge=.1, minimum_dates=100):
    gain = np.zeros(prediction.shape[1:], np.float32)
    for h in range(prediction.shape[1]):
        for group in np.unique(groups):
            members = groups == group
            x, y = prediction[:, h, members], residual[:, h, members]
            good = mask[:, h, members] & np.isfinite(x) & np.isfinite(y)
            n = good.sum(1, keepdims=True)
            if (n[:, 0] > 0).sum() < minimum_dates:
                continue
            w = good/np.maximum(n, 1)
            numerator = np.sum(w*np.where(good, x*y, 0))
            denominator = (1+ridge)*np.sum(w*np.where(good, x*x, 0))
            value = np.clip(numerator/denominator, 0., 1.) if denominator > 1e-12 else 0.
            gain[h, members] = value
    return gain
