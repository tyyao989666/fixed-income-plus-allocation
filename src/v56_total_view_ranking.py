"""Process-scoped F3 ranking alignment; frozen upstream engines stay untouched."""
from contextlib import contextmanager
import inspect

import numpy as np
import torch


def full_view_pair(pred, residual, raw_target, scale, masks):
    """Recover the total view using TRAIN-OOF baseline, never an in-sample refit.

    raw_target - residual*scale = OOF_baseline + residual_center.
    Therefore p*scale + raw_target - residual*scale is the deployed-style total
    view in the original objective units. It is used only inside training loss.
    """
    full_pred, full_target = {}, {}
    for task in pred:
        good = masks[task].bool()
        y = torch.as_tensor(raw_target[task], dtype=pred[task].dtype, device=pred[task].device)
        s = torch.as_tensor(scale[task], dtype=pred[task].dtype, device=pred[task].device)
        if y.shape != pred[task].shape or residual[task].shape != y.shape:
            raise ValueError("Total-view ranking shape mismatch")
        if not torch.isfinite(y[good]).all() or not torch.isfinite(s).all() or not (s > 0).all():
            raise ValueError("Nonfinite target or invalid residual scale")
        p = (pred[task]-residual[task])*s + y
        full_pred[task] = torch.where(good, p, torch.zeros_like(p))
        full_target[task] = torch.where(good, y, torch.zeros_like(y))
    return full_pred, full_target


@contextmanager
def total_view_ranking_implementation(engine):
    """Only F3 pairwise loss changes; F0/F1/F2, Huber and gates are unchanged.

    Serial seed execution is required, as in the mask-safe encoder context.
    The current training batch's dates are explicitly captured and checked.
    """
    original_train = engine.train_network
    signature = inspect.signature(original_train)

    def train(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        b = bound.arguments
        if b["stage"] != "F3":
            return original_train(*args, **kwargs)
        if b["cfg"]["seed_workers"] != 1:
            raise ValueError("V56 ranking context requires serial seed workers")
        arrays, split, scales = b["a"], b["split"], b["scale"]
        allowed = set(map(int, split["train"]))
        prior_batch, prior_loss = engine.batch_at, engine.group_loss
        current = {}

        def batch(a, indices, cfg, device):
            current["indices"] = np.asarray(indices, dtype=int).copy()
            return prior_batch(a, indices, cfg, device)

        def loss(pred, target, masks, modules, weights, pairwise=False):
            if not pairwise:
                return prior_loss(pred, target, masks, modules, weights, False)
            ix = current.get("indices")
            if ix is None or len(ix) != pred["asset"].shape[0] or any(int(i) not in allowed for i in ix):
                raise ValueError("F3 pairwise loss requested outside captured training batch")
            raw = {task: arrays[task+"_target"][ix] for task in pred}
            total_pred, total_target = full_view_pair(pred, target, raw, scales, masks)
            return prior_loss(total_pred, total_target, masks, modules, weights, True)

        engine.batch_at, engine.group_loss = batch, loss
        try:
            return original_train(*args, **kwargs)
        finally:
            engine.batch_at, engine.group_loss = prior_batch, prior_loss

    engine.train_network = train
    try:
        yield
    finally:
        engine.train_network = original_train
