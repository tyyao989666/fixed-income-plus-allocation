"""Train-only shared scaling preserves within-group raw-return rankings."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fit_shared_scale(target, mask, groups):
    target, mask, groups = np.asarray(target), np.asarray(mask, bool), np.asarray(groups)
    center = np.zeros(target.shape[1:], np.float32)
    scale = np.ones_like(center)
    available = np.zeros_like(center, bool)
    for h in range(target.shape[1]):
        for group in np.unique(groups):
            members = groups == group
            good = mask[:, h, members] & np.isfinite(target[:, h, members])
            values = target[:, h, members][good]
            if len(values):
                center[h, members] = values.mean()
                scale[h, members] = max(float(values.std()), 1e-4)
                available[h, members] = True
    return center, scale, available


class RawMeanHead(nn.Module):
    def __init__(self, width, kind, dropout=.4):
        super().__init__()
        if kind not in ("linear", "regularized"):
            raise ValueError("Unregistered candidate")
        self.shared = nn.LayerNorm(width) if kind == "linear" else nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 24), nn.GELU(), nn.Dropout(dropout))
        size = width if kind == "linear" else 24
        self.mean = nn.Linear(size, 3)
        self.quantiles = nn.Linear(size, 9)

    def forward(self, x):
        x = self.shared(x)
        mean = self.mean(x).transpose(1, 2)
        p = self.quantiles(x).reshape(*x.shape[:-1], 3, 3).transpose(1, 2)
        mid = p[..., 0]
        q = torch.stack([mid-F.softplus(p[..., 1]), mid, mid+F.softplus(p[..., 2])], -1)
        return mean, q


def raw_mean_loss(mean, q, target, mask, groups, cfg):
    total = mean.sum()*0
    levels = torch.tensor([.1, .5, .9], device=mean.device)
    for h, weight in enumerate([.15, .50, .35]):
        terms = []
        for group in groups.unique():
            members = groups == group
            m = mask[:, h, members].bool()
            count = m.sum(1)
            valid = count > 0
            if not valid.any():
                continue
            y = torch.where(m, target[:, h, members], 0)
            p = mean[:, h, members]
            fit = (((p-y).square()*m).sum(1)/count.clamp_min(1))[valid].mean()
            err = y[..., None]-q[:, h, members]
            pin = torch.maximum(levels*err, (levels-1)*err).mean(-1)
            pin = ((pin*m).sum(1)/count.clamp_min(1))[valid].mean()
            diff = y[:, :, None]-y[:, None, :]
            pairs = m[:, :, None] & m[:, None, :] & (diff != 0)
            pairs &= torch.triu(torch.ones_like(pairs), diagonal=1).bool()
            n = pairs.sum((1, 2))
            rank = F.softplus(-diff.sign()*(p[:, :, None]-p[:, None, :]))
            ranked = ((rank*pairs).sum((1, 2))/n.clamp_min(1))[n > 0].mean() if (n > 0).any() else fit*0
            terms.append(cfg["mean_weight"]*fit+cfg["pinball_weight"]*pin+cfg["rank_weight"]*ranked)
        if terms:
            total = total+weight*torch.stack(terms).mean()
    return total


def ep_mean_view(payload):
    if payload.get("units") != "cumulative_raw_relative_price_return":
        raise ValueError("EP mean view must use cumulative raw relative price returns")
    if payload.get("horizon_days") not in (5, 20, 60):
        raise ValueError("Unknown horizon; no implicit annualization")
    if np.datetime64(payload["available_after"]) <= np.datetime64(payload["as_of"]):
        raise ValueError("T+1 availability required")
    mean = np.asarray(payload["mean"], float)
    if not np.isfinite(mean).all():
        raise ValueError("Nonfinite mean")
    if payload.get("approved") is not True or payload.get("execution_certified") is not True:
        raise ValueError("Research forecast is not approved for formal EP execution")
    return mean
