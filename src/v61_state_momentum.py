"""Causal momentum cohorts, explicit missing Boundary and rank-to-return calibration."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


def monthly_cohort_votes(returns, months=(1, 3, 6, 9, 12)):
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("Sorted unique dates required")
    complete = []
    for period, f in returns.groupby(returns.index.to_period("M")):
        if period.end_time.normalize() > returns.index[-1]:
            continue
        good = f.notna().all(0) & (f > -1).all(0)
        value = np.expm1(np.log1p(f).sum(0)).where(good)
        complete.append(pd.DataFrame([value], index=[f.index[-1]]))
    if not complete:
        return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    monthly = pd.concat(complete)
    components = []
    for lookback in months:
        momentum = np.expm1(np.log1p(monthly).rolling(lookback, min_periods=lookback).sum())
        signal = (np.sign(momentum)+1)/2
        for holding in months:
            components.append(signal.rolling(holding, min_periods=holding).mean())
    stack = np.stack([f.to_numpy() for f in components])
    valid = np.isfinite(stack).all(0)
    votes = pd.DataFrame(np.where(valid, np.nan_to_num(stack).mean(0), np.nan), index=monthly.index, columns=returns.columns)
    # State known at each month-end close; next-session execution is separate.
    daily = votes.reindex(returns.index, method="ffill")
    return daily.where(returns.notna())


def price_rule_signals(returns, modules, threshold=2.):
    monthly = monthly_cohort_votes(returns)
    momentum20 = np.expm1(np.log1p(returns).rolling(20, min_periods=20).sum())
    annual = np.expm1(np.log1p(returns).rolling(252, min_periods=252).sum())
    vol = returns.rolling(20, min_periods=20).std(ddof=0)*np.sqrt(20)
    extension = momentum20/vol.where(vol > 1e-8)
    apply = np.isin(modules, [2, 3])[None, :]
    hot = (extension > threshold) & (annual > .10) & apply
    softened = monthly.where(~hot, .5+.5*(monthly-.5))
    return dict(momentum20=momentum20, cohort=monthly, price_soft=softened, extension=extension,
                value_boundary=pd.DataFrame(np.nan, index=returns.index, columns=returns.columns))


def nested_split(a, split, days=126, minimum=300):
    train = split["train"]
    calibration = train[-days:]
    dates = a["dates"]
    from src.v45_forecast import end_times
    ends = end_times(a)
    inner = train[(ends[train] > 0).all(1) & (ends[train].max(1) < dates[calibration[0]])]
    if len(inner) < minimum:
        raise ValueError("Insufficient inner train after calibration and label purge")
    if not (ends[calibration].max(1) < dates[split["validation"][0]]).all():
        raise ValueError("Calibration labels overlap validation observations")
    return dict(train=inner, calibration=calibration, validation=split["validation"])


class StateRanker(nn.Module):
    def __init__(self, forecast_network, width):
        super().__init__()
        self.network = forecast_network
        for p in self.network.parameters():
            p.requires_grad_(False)
        for p in self.network.core.parameters():
            p.requires_grad_(True)
        self.heads = nn.ModuleDict({t:nn.Linear(width, 3) for t in ["asset", "sleeve"]})
        for h in self.heads.values():
            nn.init.normal_(h.weight, std=.01)
            nn.init.zeros_(h.bias)

    def forward(self, batch):
        assets, _, _, sleeves = self.network.encode(batch)
        return {t:self.heads[t](v).transpose(1, 2) for t,v in [("asset", assets), ("sleeve", sleeves)]}


def rank_loss(pred, targets, masks, groups, penalty=.001):
    total = pred["asset"].sum()*0
    for task in pred:
        for h,w in enumerate([.15, .50, .35]):
            terms = []
            for group in groups[task].unique():
                ix = groups[task] == group
                p = pred[task][:, h, ix]
                mask = masks[task][:, h, ix].bool()
                y = torch.where(mask, targets[task][:, h, ix], 0)
                diff = y[:, :, None]-y[:, None, :]
                pairs = mask[:, :, None] & mask[:, None, :] & (diff.abs() > 1e-8)
                pairs &= torch.triu(torch.ones_like(pairs), diagonal=1).bool()
                n = pairs.sum((1, 2))
                if not (n > 0).any():
                    continue
                error = F.softplus(-diff.sign()*(p[:, :, None]-p[:, None, :]))
                loss = ((error*pairs).sum((1, 2))/n.clamp_min(1))[n > 0].mean()
                count = mask.sum(1)
                norm = ((p.square()*mask).sum(1)/count.clamp_min(1))[count > 0].mean()
                terms.append(loss+penalty*norm)
            if terms:
                total = total+.5*w*torch.stack(terms).mean()
    return total


def fit_calibrator(scores, truth, mask, prior, groups, ridge=.1, minimum_dates=100):
    slopes = np.zeros(prior.shape, np.float32)
    for h in range(3):
        for group in np.unique(groups):
            ix = groups == group
            x, y = scores[:, h, ix], truth[:, h, ix]-prior[h, ix]
            good = mask[:, h, ix] & np.isfinite(x) & np.isfinite(y)
            n = good.sum(1, keepdims=True)
            if int((n[:, 0] >= 2).sum()) < minimum_dates:
                continue
            weights = good/np.maximum(n, 1)
            numerator = np.sum(weights*np.where(good, x*y, 0))
            denominator = (1+ridge)*np.sum(weights*np.where(good, x*x, 0))
            beta = max(numerator/denominator, 0.) if denominator > 1e-12 else 0.
            slopes[h, ix] = beta
    return slopes
