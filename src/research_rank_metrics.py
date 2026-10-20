"""Explicit reference identity for the legacy V61 rank aggregation convention."""
import numpy as np
from src import v45_forecast as engine


def rank_metrics_named(pred, baselines, truth, masks, groups, reference):
    if reference not in baselines:
        raise ValueError('Explicit reference baseline is missing')
    if not baselines or set(pred) != set(truth) or set(pred) != set(masks):
        raise ValueError('Incompatible prediction tasks')

    def ranked(p, y, m):
        value = engine.daily_ic(p, y, m)
        good = m & np.isfinite(p) & np.isfinite(y)
        for i in range(len(value)):
            if good[i].sum() >= 2 and np.ptp(y[i, good[i]]) > 0 and np.ptp(p[i, good[i]]) == 0:
                value[i] = 0.
        return value

    deltas, absolute = {}, []
    for name, base in baselines.items():
        terms = []
        for task in pred:
            for h in range(pred[task].shape[1]):
                group_delta, group_ic = [], []
                for group in np.unique(groups[task]):
                    members = groups[task] == group
                    m = masks[task][:, h, members]
                    p, y, b = [value[task][:, h, members] for value in [pred, truth, base]]
                    # Pairwise finite support must be shared by both sides.
                    m = m & np.isfinite(p) & np.isfinite(y) & np.isfinite(b)
                    x, z = ranked(p, y, m), ranked(b, y, m)
                    group_delta.append(engine.finite_mean(x-z))
                    group_ic.append(engine.finite_mean(x))
                terms.append(engine.finite_mean(group_delta))
                if name == reference:
                    absolute.extend(group_ic)
        deltas[name] = engine.finite_mean(terms)
    minimum = min(deltas.values()) if np.isfinite(list(deltas.values())).all() else -float('inf')
    return minimum, deltas, engine.finite_mean(absolute)
