"""Mask-safe research encoder; old v45 checkpoint semantics remain unchanged."""
from __future__ import annotations

import hashlib
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch

from src import v45_forecast as legacy
from src.hierarchical_asset_itransformer import HierarchicalAssetITransformer

_original_batch_at = legacy.batch_at


def masked_pool(values, logits, valid, dim):
    valid = valid.bool()
    # Empty groups yield zero, not softmax(-inf, ..., -inf) -> NaN.
    score = logits.masked_fill(~valid, -torch.inf)
    score = torch.where(valid.any(dim=dim, keepdim=True), score, torch.zeros_like(score))
    weight = torch.softmax(score, dim=dim)*valid.to(values.dtype)
    return (values*weight.unsqueeze(-1)).sum(dim=dim)


class MaskSafeCore(HierarchicalAssetITransformer):
    def encode_context(self, local_values, local_masks, macro_values):
        if local_values.shape != local_masks.shape or local_values.ndim != 4:
            raise ValueError("Local values/masks must have matching [B,L,A,F] shapes")
        if local_values.shape[1:] != (self.sequence_length, self.asset_count, self.local_features):
            raise ValueError("Local feature schema mismatch")
        if not torch.all((local_masks == 0) | (local_masks == 1)):
            raise ValueError("Nonbinary local masks")
        observed = local_masks.bool()
        if not torch.isfinite(local_values[observed]).all() or not torch.isfinite(macro_values).all():
            raise ValueError("Nonfinite observed encoder input")
        x = torch.where(observed, local_values, torch.zeros_like(local_values))
        batch = len(x)
        asset_valid = observed.any(dim=1).any(dim=-1)
        feature_valid = observed.any(dim=1)
        local_valid = torch.cat([feature_valid, asset_valid[:, :, None].expand_as(feature_valid)], -1)
        local_valid = local_valid.reshape(batch*self.asset_count, self.local_token_count)
        raw = torch.cat([x, local_masks], -1).permute(0, 2, 3, 1)
        local = self.local_projection(raw.reshape(batch*self.asset_count, self.local_token_count, self.sequence_length))
        ids = torch.arange(self.local_token_count, device=x.device)
        local = local+self.local_feature_embedding(ids)[None]
        if not isinstance(self.local_encoder, torch.nn.Identity):
            padding = ~local_valid.clone()
            padding[~local_valid.any(1), 0] = False
            local = self.local_encoder(local, src_key_padding_mask=padding)
        asset = masked_pool(local, self.local_pool_score(local).squeeze(-1), local_valid, 1)
        asset = asset.reshape(batch, self.asset_count, -1)
        asset_ids = torch.arange(self.asset_count, device=x.device)
        asset = (asset+self.asset_embedding(asset_ids)[None]+self.module_embedding(self.module_index)[None]
                 +self.token_type(torch.zeros(self.asset_count, dtype=torch.long, device=x.device))[None])
        macro = self.macro_projection(macro_values.permute(0, 2, 1))
        mid = torch.arange(self.macro_features, device=x.device)
        macro = macro+self.macro_feature_embedding(mid)[None]+self.token_type(torch.ones(self.macro_features, dtype=torch.long, device=x.device))[None]
        # Macro availability tokens explicitly encode missingness and are known
        # observations, including when every macro value itself is missing.
        padding = torch.cat([~asset_valid, torch.zeros(batch, self.macro_features, dtype=torch.bool, device=x.device)], 1)
        encoded = self.encoder(torch.cat([asset, macro], 1), src_key_padding_mask=padding)
        assets = encoded[:, :self.asset_count]*asset_valid[..., None]
        macros = encoded[:, self.asset_count:]
        macro_pool = masked_pool(macros, self.macro_pool_score(macros).squeeze(-1), ~padding[:, self.asset_count:], 1)
        sleeves = []
        for group in range(5):
            members = self.module_index == group
            count = asset_valid[:, members].sum(1, keepdim=True)
            pooled = assets[:, members].sum(1)/count.clamp_min(1)
            value = self.sleeve_head(torch.cat([pooled, macro_pool], 1))
            sleeves.append(value*(count > 0))
        return assets, macros, macro_pool, torch.stack(sleeves, 1)


class MaskSafeForecastNetwork(legacy.ForecastNetwork):
    def __init__(self, arrays, cfg):
        super().__init__(arrays, cfg)
        macro = arrays["macro_values"].shape[1]*(2 if "macro_masks" in arrays else 1)
        self.core = MaskSafeCore(cfg["sequence_length"], arrays["local_values"].shape[2],
                                 max(macro, 1), arrays["module_index"], 3, cfg)


def validate_dataset(a):
    legacy.validate_arrays(a)
    t, n, _ = a["local_values"].shape
    for key in ["local_masks", "return_masks", "asset_target_mask", "sleeve_target_mask"]:
        if not np.isin(a[key], [0, 1]).all():
            raise ValueError(f"Nonbinary {key}")
    if a["returns"].shape != (t, n) or a["return_masks"].shape != (t, n):
        raise ValueError("Return schema mismatch")
    good = a["return_masks"].astype(bool)
    if not np.isfinite(a["returns"][good]).all() or (a["returns"][good] <= -1).any():
        raise ValueError("Invalid observed returns")
    local = a["local_masks"].astype(bool)
    if not np.isfinite(a["local_values"][local]).all():
        raise ValueError("Invalid observed local values")
    if "macro_masks" in a:
        if a["macro_masks"].shape != a["macro_values"].shape or not np.isin(a["macro_masks"], [0, 1]).all():
            raise ValueError("Invalid macro mask schema")
    if a["macro_values"].shape[0] != t or len(a["module_index"]) != n:
        raise ValueError("Feature/entity schema mismatch")
    dates, ends = legacy.dates_ns(a["dates"]), legacy.end_times(a)
    for j, horizon in enumerate([5, 20, 60]):
        expected = dates[horizon:]
        if not np.array_equal(ends[:-horizon, j], expected):
            raise ValueError(f"Wrong {horizon}-day label endpoint")
        for task in ["asset", "sleeve"]:
            if a[f"{task}_target_mask"][-horizon:, j].any():
                raise ValueError(f"Unmatured {task} labels exposed as valid")


def safe_batch_at(a, indices, cfg, device):
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("Indices must be a nonempty integer vector")
    if (indices < cfg["sequence_length"]-1).any() or (indices >= len(a["dates"])).any():
        raise ValueError("Window index outside available historical observations")
    # Values under false masks are never exposed, even if a corrupt source
    # contains NaN or an arbitrary placeholder in those cells.
    x, mask, macro = _original_batch_at(a, indices, cfg, device)
    if "macro_masks" in a:
        f = a["macro_values"].shape[1]
        macro = torch.cat([torch.where(macro[..., f:].bool(), macro[..., :f], torch.zeros_like(macro[..., :f])), macro[..., f:]], -1)
    return x, mask, macro


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@contextmanager
def training_implementation():
    """Opt-in new research training only; no permanent change to legacy globals.

    Use seed_workers=1 inside this context. Spawned workers import fresh legacy
    globals and would otherwise silently train the wrong encoder implementation.
    """
    prior_network, prior_batch = legacy.ForecastNetwork, legacy.batch_at
    prior_workers = legacy.seed_worker_results
    def serial_results(payload, seeds, workers):
        if workers != 1:
            raise ValueError("Mask-safe training requires seed_workers=1; spawned legacy workers are forbidden")
        yield from prior_workers(payload, seeds, workers)
    legacy.ForecastNetwork, legacy.batch_at = MaskSafeForecastNetwork, safe_batch_at
    legacy.seed_worker_results = serial_results
    try:
        yield
    finally:
        legacy.ForecastNetwork, legacy.batch_at = prior_network, prior_batch
        legacy.seed_worker_results = prior_workers
