"""Rebuild research features and causal targets after an identity-safe replacement."""
import numpy as np
import pandas as pd

from src.v45_data import local_features, observable_grid, reference_weights, label_reference_policy
from src.v53_unified_data import independent_cb_relative_labels, validate_relative_dataset
from src.v55_learning_objective import temporal_opportunity_targets, decode_opportunity


def exact_calendar_returns(levels, dates, identifier):
    required = {"date", "close", "source_identifier"}
    if required - set(levels):
        raise ValueError("Source identity, dates and closes required")
    if levels.empty or not levels.source_identifier.eq(identifier).all():
        raise ValueError("Wrong or ambiguous source identity")
    frame = levels.copy()
    frame.date = pd.to_datetime(frame.date, errors="raise")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError("Ordered unique source dates required")
    close = frame.set_index("date").close.reindex(dates)
    if not np.isfinite(close).all() or not close.gt(0).all():
        raise ValueError("Complete positive source calendar required; no forward fill")
    return close.pct_change(fill_method=None)


def rebuild(arrays, returns, bars, metadata, cb_reference):
    """Keep original task/benchmarks; do not fit anything on observed test returns."""
    result = {k: v.copy() for k, v in arrays.items()}
    assert list(returns.columns) == metadata["assets"]
    assert np.array_equal(pd.DatetimeIndex(returns.index).asi8, result["dates"])
    result["returns"] = returns.fillna(0).to_numpy(dtype=np.float32)
    result["return_masks"] = returns.notna().to_numpy()
    result["local_values"], result["local_masks"], names = local_features(returns, bars)
    result["macro_values"], result["macro_masks"], result["regime_index"] = observable_grid(returns)
    preferred, _ = label_reference_policy(metadata["assets"], metadata["asset_modules"])
    sw = reference_weights(result["returns"], result["return_masks"], preferred, result["module_index"])
    result["sleeve_reference_weights"] = sw
    result.update(independent_cb_relative_labels(result["dates"], result["returns"], result["return_masks"],
        result["module_index"], sw, result["balanced_reference_weights"], cb_reference))
    sr = np.sum(sw * result["returns"][:, None, :], axis=-1)
    sm = (sw.sum(-1) > 0) & ((sw == 0) | result["return_masks"][:, None, :]).all(-1)
    sr[:, 2], sm[:, 2] = cb_reference, np.isfinite(cb_reference)
    for task, daily in (("asset", returns.to_numpy()), ("sleeve", np.where(sm, sr, np.nan))):
        y, mask = result[task + "_target"].copy(), result[task + "_target_mask"].copy()
        transformed = temporal_opportunity_targets(y, mask, daily)
        result[task + "_raw_relative_target"], result[task + "_raw_relative_target_mask"] = y, mask
        for suffix, key in (("_target", "target"), ("_target_mask", "mask"),
                            ("_opportunity_scale", "scale"), ("_opportunity_offset", "offset")):
            result[task + suffix] = transformed[key]
        good = transformed["mask"]
        assert np.allclose(decode_opportunity(transformed["target"], transformed["scale"],
            transformed["offset"])[good], y[good], atol=1e-6)
    validate_relative_dataset(result)
    return result, names
