"""Explicit zero-increment candidate for future E-only model selection."""
import numpy as np


def zero_or_neural(early_MSE_ratio, selected_epoch, valid_dates, minimum_dates=100,
                   minimum_improvement=1e-4):
    if minimum_dates < 1 or minimum_improvement < 0:
        raise ValueError('Invalid predeclared qualification settings')
    if not np.isfinite(valid_dates) or valid_dates < minimum_dates or valid_dates != int(valid_dates):
        reason = 'insufficient_inner_E_support'
    elif not np.isfinite(early_MSE_ratio):
        reason = 'nonfinite_inner_E_metric'
    elif not np.isfinite(selected_epoch) or selected_epoch < 1 or selected_epoch != int(selected_epoch):
        reason = 'no_trained_checkpoint'
    elif early_MSE_ratio >= 1.-minimum_improvement:
        reason = 'zero_increment_not_beaten_on_E'
    else:
        return dict(neural_enabled=True, candidate='neural', reason='inner_E_increment_supported',
                    qualification_source='inner_E_only', formal_promotion=False)
    return dict(neural_enabled=False, candidate='zero_increment', reason=reason,
                qualification_source='inner_E_only', formal_promotion=False)
