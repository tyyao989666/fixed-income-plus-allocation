"""Mode-specific losses without changing inactive prediction heads."""
import numpy as np
from src.v63_residual_learning import residual_loss


def active_tasks(mode):
    if mode == 'joint':
        return ('asset', 'sleeve')
    if mode == 'asset_only':
        return ('asset',)
    if mode == 'sleeve_only':
        return ('sleeve',)
    raise ValueError('Unregistered task ablation mode')


def mode_loss(pred, targets, masks, groups, mode, rank_weight=.05, penalty=.01):
    active = active_tasks(mode)
    use = {t: masks[t] if t in active else masks[t] & False for t in masks}
    return residual_loss(pred, targets, use, groups, rank_weight, penalty)*(2/len(active))


def split_inner(train, ends, dates, validation_days=126, minimum_fit_days=300):
    if len(train) <= validation_days:
        return None
    validation = train[-validation_days:]
    fit = train[(ends[train] > 0).all(1) & (ends[train].max(1) < dates[validation[0]])]
    if len(fit) < minimum_fit_days:
        return None
    return dict(fit=fit, early=validation)


def activate_heads(model, mode):
    active = active_tasks(mode)
    for task, head in model.heads.items():
        for p in head.parameters():
            p.requires_grad_(task in active)
    return [p for task, head in model.heads.items() if task in active for p in head.parameters()]
