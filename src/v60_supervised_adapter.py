"""Exact frozen-prefix replay plus supervised final iTransformer block."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.v49_masked_forecast import masked_pool
from src.v59_raw_mean import fit_shared_scale


def fit_prior(y, mask, groups, shrinkage=126):
    center, scale, group_available = fit_shared_scale(y, mask, groups)
    m = np.asarray(mask, bool) & np.isfinite(y)
    count = m.sum(0)
    mean = np.divide(np.where(m, y, 0).sum(0), count, out=center.astype(float).copy(), where=count > 0)
    weight = count/(count+shrinkage)
    prior = (weight*mean+(1-weight)*center).astype(np.float32)
    return prior, center, scale, group_available, count


@torch.no_grad()
def capture_prefix(network, batch):
    network.eval()
    captured = {}

    def hook(module, args, kwargs):
        captured["tokens"] = args[0].detach()
        mask = kwargs.get("src_key_padding_mask")
        if mask is None:
            raise ValueError("Expected explicit source padding")
        captured["padding"] = mask.detach()

    handle = network.core.encoder.layers[-1].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        full = network.encode(batch)
    finally:
        handle.remove()
    if "tokens" not in captured:
        raise ValueError("Last block hook bypassed")
    captured["asset_valid"] = batch[1].bool().any(1).any(-1)
    return captured, {"asset": full[0], "sleeve": full[3]}


class SupervisedAdapter(nn.Module):
    def __init__(self, core, priors, centers, scales, mode, bound=1.):
        super().__init__()
        if mode not in ("frozen_control", "last_block_adapter"):
            raise ValueError("Unregistered mode")
        self.core = copy.deepcopy(core)
        for p in self.core.parameters():
            p.requires_grad_(False)
        self.mode, self.bound = mode, bound
        if mode == "last_block_adapter":
            for p in self.core.encoder.layers[-1].parameters():
                p.requires_grad_(True)
        width = self.core.asset_embedding.embedding_dim
        self.heads = nn.ModuleDict({t: nn.Linear(width, 3) for t in ["asset", "sleeve"]})
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for task in self.heads:
            self.register_buffer(task+"_prior", torch.as_tensor(priors[task]))
            self.register_buffer(task+"_center", torch.as_tensor(centers[task]))
            self.register_buffer(task+"_scale", torch.as_tensor(scales[task]))

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        if self.mode == "last_block_adapter":
            self.core.encoder.layers[-1].train(mode)
        return self

    def representations(self, prefix):
        encoded = self.core.encoder.layers[-1](prefix["tokens"], src_key_padding_mask=prefix["padding"])
        if self.core.encoder.norm is not None:
            encoded = self.core.encoder.norm(encoded)
        valid = prefix["asset_valid"]
        assets = encoded[:, :self.core.asset_count]*valid[..., None]
        macros = encoded[:, self.core.asset_count:]
        macro_valid = torch.ones(macros.shape[:2], device=macros.device, dtype=torch.bool)
        pool = masked_pool(macros, self.core.macro_pool_score(macros).squeeze(-1), macro_valid, 1)
        sleeves = []
        for group in range(5):
            members = self.core.module_index == group
            count = valid[:, members].sum(1, keepdim=True)
            avg = assets[:, members].sum(1)/count.clamp_min(1)
            value = self.core.sleeve_head(torch.cat([avg, pool], 1))
            sleeves.append(value*(count > 0))
        return dict(asset=assets, sleeve=torch.stack(sleeves, 1))

    def forward(self, prefix):
        features = self.representations(prefix)
        normalized, increments = {}, {}
        for task in self.heads:
            delta = self.bound*torch.tanh(self.heads[task](features[task]).transpose(1, 2))
            prior = getattr(self, task+"_prior")
            center = getattr(self, task+"_center")
            scale = getattr(self, task+"_scale")
            normalized[task] = (prior-center)/scale+delta
            increments[task] = delta
        return normalized, increments

    def decode(self, pred):
        return {t: p*getattr(self, t+"_scale")+getattr(self, t+"_center") for t,p in pred.items()}


def supervised_loss(pred, increments, targets, masks, groups, cfg):
    total = pred["asset"].sum()*0
    for task in pred:
        for h, weight in enumerate([.15, .50, .35]):
            terms = []
            for group in groups[task].unique():
                ix = groups[task] == group
                m = masks[task][:, h, ix].bool()
                count = m.sum(1)
                valid = count > 0
                if not valid.any():
                    continue
                p = pred[task][:, h, ix]
                y = torch.where(m, targets[task][:, h, ix], 0)
                fit = (((p-y).square()*m).sum(1)/count.clamp_min(1))[valid].mean()
                delta = increments[task][:, h, ix]
                penalty = ((delta.square()*m).sum(1)/count.clamp_min(1))[valid].mean()
                diff = y[:, :, None]-y[:, None, :]
                pairs = m[:, :, None] & m[:, None, :] & (diff != 0)
                pairs &= torch.triu(torch.ones_like(pairs), diagonal=1).bool()
                n = pairs.sum((1, 2))
                error = F.softplus(-diff.sign()*(p[:, :, None]-p[:, None, :]))
                ranking = ((error*pairs).sum((1, 2))/n.clamp_min(1))[n > 0].mean() if (n > 0).any() else fit*0
                terms.append(fit+cfg["rank_weight"]*ranking+cfg["residual_penalty"]*penalty)
            if terms:
                total = total+.5*weight*torch.stack(terms).mean()
    return total
