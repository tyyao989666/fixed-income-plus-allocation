"""Online maturity-only estimates with hierarchical nine-grid shrinkage."""
import numpy as np


def _mean_ess(values, observed, weights):
    w = weights[:, None]*observed
    total = w.sum(0)
    mean = np.divide((w*np.where(observed, values, 0)).sum(0), total,
                     out=np.zeros(values.shape[1]), where=total > 0)
    ess = np.divide(total*total, (w*w).sum(0), out=np.zeros_like(total), where=(w*w).sum(0) > 0)
    return mean, ess


def _shrink(mean, ess, parent, strength):
    weight = ess/(ess+strength)
    return weight*mean+(1-weight)*parent


def online_grid_control(dates, ends, truth, mask, groups, regimes, sources, queries,
                        history_sessions=756, half_life_sessions=126,
                        minimum_group_dates=60, shrinkage=126):
    dates, ends, truth = np.asarray(dates), np.asarray(ends), np.asarray(truth)
    mask, groups, regimes = np.asarray(mask, bool), np.asarray(groups), np.asarray(regimes)
    sources, queries = np.asarray(sources, int), np.asarray(queries, int)
    if truth.shape != mask.shape or truth.ndim != 3 or ends.shape != truth.shape[:2]:
        raise ValueError('Target/maturity shape mismatch')
    if len(dates) != len(truth) or len(regimes) != len(dates) or len(groups) != truth.shape[-1]:
        raise ValueError('Calendar/group shape mismatch')
    if np.any(np.diff(dates) <= 0) or np.any(np.diff(sources) <= 0):
        raise ValueError('Ordered unique source/calendar required')
    if np.any(sources < 0) or np.any(sources >= len(dates)) or np.any(queries < 0) or np.any(queries >= len(dates)):
        raise ValueError('Index outside calendar')
    if half_life_sessions <= 0 or history_sessions < 1 or minimum_group_dates < 1 or shrinkage <= 0:
        raise ValueError('Invalid causal control parameters')
    if not np.isin(regimes, np.arange(-1, 9)).all():
        raise ValueError('Unknown nine-grid encoding')
    shape = (len(queries), *truth.shape[1:])
    mean, conditional = np.full(shape, np.nan), np.full(shape, np.nan)
    joint_ess, counts = np.zeros(shape), np.zeros(shape, np.int32)
    latest = np.zeros(shape[:2], np.int64)
    mean_age = np.full(shape[:2], np.nan)
    for position, query in enumerate(queries):
        for h in range(truth.shape[1]):
            past = sources[(sources < query) & (sources >= query-history_sessions)
                & (ends[sources, h] > 0) & (ends[sources, h] < dates[query])]
            if not len(past):
                continue
            values = truth[past, h]
            good = mask[past, h] & np.isfinite(values)
            maturity_positions = np.searchsorted(dates, ends[past, h], side='left')
            age = query-maturity_positions
            if (age <= 0).any():
                raise AssertionError('Immature information entered control')
            weights = np.exp2(-age/half_life_sessions)
            used = good.any(1)
            if used.any():
                latest[position, h] = ends[past[used], h].max()
                mean_age[position, h] = np.average(age[used], weights=weights[used])
            for group in np.unique(groups):
                members = groups == group
                y, m = values[:, members], good[:, members]
                n = m.sum(1)
                usable = n > 0
                if usable.sum() < minimum_group_dates:
                    continue
                daily = np.where(m, y, 0).sum(1)/np.maximum(n, 1)
                pooled = np.average(daily[usable], weights=weights[usable])
                own_mean, own_ess = _mean_ess(y, m, weights)
                global_entity = _shrink(own_mean, own_ess, pooled, shrinkage)
                mean[position, h, members] = global_entity
                counts[position, h, members] = m.sum(0)
                current = regimes[query]
                if current < 0:
                    conditional[position, h, members] = global_entity
                    continue
                past_states = regimes[past]
                known = past_states >= 0
                bond = known & (past_states//3 == current//3)
                equity = known & (past_states % 3 == current % 3)
                axis = []
                for match in [bond, equity]:
                    v, ess = _mean_ess(y, m & match[:, None], weights)
                    axis.append(_shrink(v, ess, global_entity, shrinkage))
                parent = .5*(axis[0]+axis[1])
                v, ess = _mean_ess(y, m & (past_states == current)[:, None], weights)
                conditional[position, h, members] = _shrink(v, ess, parent, shrinkage)
                joint_ess[position, h, members] = ess
    assert np.all((latest == 0) | (latest < dates[queries, None]))
    return dict(online_mean=mean.astype(np.float32), online_grid=conditional.astype(np.float32),
        latest_maturity=latest, joint_weight_ESS=joint_ess.astype(np.float32),
        history_counts=counts, mean_information_age_sessions=mean_age)
