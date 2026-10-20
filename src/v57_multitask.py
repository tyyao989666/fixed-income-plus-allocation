"""Distributional iTransformer transfer heads and fail-closed execution contract."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def future_downside(returns, observed, horizons=(5, 20, 60)):
    """Realized forward maximum drawdown, including initial wealth=1.

    These are LABELS only. A single missing return invalidates that cell.
    The last h dates are never treated as matured observations.
    """
    r, m = np.asarray(returns, float), np.asarray(observed, bool)
    values = np.zeros((len(r), len(horizons), r.shape[1]), np.float32)
    masks = np.zeros_like(values, bool)
    for j, h in enumerate(horizons):
        for t in range(len(r)-h):
            block = r[t+1:t+h+1]
            good = m[t+1:t+h+1].all(0) & np.isfinite(block).all(0) & (block > -1).all(0)
            path = np.vstack([np.ones(r.shape[1]), np.cumprod(1+np.where(np.isfinite(block), block, 0), axis=0)])
            dd = -np.min(path/np.maximum.accumulate(path, axis=0)-1, axis=0)
            values[t,j] = np.where(good, dd, 0)
            masks[t,j] = good
    return values, masks


class DistributionHead(nn.Module):
    def __init__(self, width=40):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.GELU(),
                                    nn.Dropout(.20), nn.Linear(48, 15))

    def forward(self, features):
        p = self.layers(features).reshape(*features.shape[:-1], 3, 5).transpose(1,2)
        median = p[...,0]
        quantiles = torch.stack([median-F.softplus(p[...,1]), median,
                                 median+F.softplus(p[...,2])], -1)
        return quantiles, F.softplus(p[...,3]), p[...,4]


def multitask_loss(q, risk, mean, target, target_mask, downside, downside_mask, modules):
    """Equal sleeve/date weighting, not pooled bond-cell domination."""
    total = q.sum()*0
    levels = torch.tensor([.1,.5,.9], device=q.device)
    for j, weight in enumerate([.15,.50,.35]):
        terms = []
        for group in modules.unique():
            ix = modules == group
            mask = target_mask[:,j,ix].bool()
            y = torch.where(mask, target[:,j,ix], 0)
            pred = q[:,j,ix]
            error = y[...,None]-pred
            pinball = torch.maximum(levels*error, (levels-1)*error).mean(-1)
            count = mask.sum(1)
            valid = count > 0
            if not valid.any():
                continue
            fit = ((pinball*mask).sum(1)/count.clamp_min(1))[valid].mean()
            mean_fit = (((mean[:,j,ix]-y).square()*mask).sum(1)/count.clamp_min(1))[valid].mean()
            diff = y[:,:,None]-y[:,None,:]
            pairs = mask[:,:,None] & mask[:,None,:] & (diff != 0)
            pairs &= torch.triu(torch.ones_like(pairs), diagonal=1).bool()
            ranked = mean[:,j,ix]
            rank = F.softplus(-diff.sign()*(ranked[:,:,None]-ranked[:,None,:]))
            n = pairs.sum((1,2))
            ranking = ((rank*pairs).sum((1,2))/n.clamp_min(1))[n>0].mean() if (n>0).any() else fit*0
            dm = downside_mask[:,j,ix].bool()
            dr = torch.where(dm, downside[:,j,ix], 0)
            nr = dm.sum(1)
            risk_error = F.smooth_l1_loss(risk[:,j,ix], dr, reduction="none")
            auxiliary = ((risk_error*dm).sum(1)/nr.clamp_min(1))[nr>0].mean() if (nr>0).any() else fit*0
            terms.append(fit+.20*ranking+.25*auxiliary+.10*mean_fit)
        if terms:
            total = total+weight*torch.stack(terms).mean()
    return total


def require_execution_certificate(certificate, dataset_hash, assets, root):
    """No guessed cash flows, unchecked boolean flag, or proxy reward promotion."""
    required = ["cashflows_complete", "execution_prices_verified", "corporate_actions_complete",
                "cost_model_verified", "independent_review_passed"]
    if not certificate or certificate.get("dataset_sha256") != dataset_hash:
        raise ValueError("Missing or mismatched execution certificate")
    if any(certificate.get(k) is not True for k in required):
        raise ValueError("Execution evidence incomplete")
    if set(certificate.get("assets", [])) != set(assets):
        raise ValueError("Certificate does not cover all requested assets")
    if not certificate.get("evidence"):
        raise ValueError("No independently reviewed source evidence")
    for item in certificate["evidence"]:
        p = (Path(root)/item["path"]).resolve()
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Execution evidence hash mismatch")


def execution_observation(median, lower, upper, downside, holdings, target, age, cost_bps):
    """Current-known raw-return views + inventory; never forward labels/rewards."""
    arrays = [np.asarray(x, float) for x in [median, lower, upper, downside, holdings, target]]
    if any(x.shape != arrays[0].shape for x in arrays) or any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("Execution observation requires aligned finite arrays")
    mid, lo, hi, risk, current, wanted = arrays
    if (lo>mid).any() or (mid>hi).any() or (risk<0).any() or age<0 or cost_bps<0:
        raise ValueError("Invalid predictive interval or execution state")
    if not np.isclose(current.sum(),1) or not np.isclose(wanted.sum(),1) or (current<0).any() or (wanted<0).any():
        raise ValueError("No-cash, fully invested long-only holdings required")
    turnover = np.abs(wanted-current).sum()
    return np.r_[mid, hi-lo, risk, current, wanted-current,
                 np.log1p(age), turnover, turnover*cost_bps/10000].astype(np.float32)
