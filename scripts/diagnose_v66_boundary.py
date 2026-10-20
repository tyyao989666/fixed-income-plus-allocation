"""One fixed F-only linear probe; exploratory, not certified trading evidence."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import v45_forecast as engine
from scripts.build_v66_boundary import OUT, DATA, SOURCE, sha


def main():
    status = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
    for name, digest in status['source_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed source: '+name)
    with np.load(SOURCE) as z:
        a = {k:z[k] for k in z.files}
    split_path = ROOT / 'data_processed/task_ablation_v64/2024.npz'
    with np.load(split_path) as z:
        r = {k:z[k] for k in z.files}
    v64_status = json.loads((ROOT / 'output/reports/task_ablation_v64/target_status.json').read_text())
    assert sha(split_path) == v64_status['artifacts'][str(split_path.relative_to(ROOT))]
    with np.load(DATA / 'research_boundary_features.npz') as z:
        x_boundary = z['values']
    daily = pd.read_csv(OUT / 'daily_boundary_state.csv')
    x = np.column_stack([daily['price_momentum_252d'], x_boundary])
    parts = {name:r[name+'_indices'] for name in ['fit', 'early', 'calibration', 'validation']}
    fit = parts['fit']
    ends = engine.end_times(a)
    assert (ends[fit] > 0).all() and (ends[fit] < a['dates'][parts['early'][0]]).all()
    finite = np.isfinite(x).all(1)
    if not finite[fit].all():
        raise ValueError('Unexpected feature gaps in fixed F; do not silently change experiment')
    mean, std = x[fit].mean(0), x[fit].std(0)
    std = np.where(std > 1e-12, std, 1)
    standardized = (x-mean)/std
    modes = dict(fit_entity_mean=[], price_momentum=[0], momentum_plus_boundary=[0, 1, 2, 3],
                 momentum_boundary_interactions=list(range(6)))
    rows, coefficients = [], []
    for task in ['asset', 'sleeve']:
        members = np.flatnonzero(a['module_index'] == 3) if task == 'asset' else np.array([3])
        y = a[task+'_raw_relative_target'][:, :, members]
        mask = a[task+'_raw_relative_target_mask'][:, :, members].astype(bool) & np.isfinite(y)
        predictions = {}
        for mode, cols in modes.items():
            design = np.column_stack([np.ones(len(x)), standardized[:, cols]])
            p = np.full(y.shape, np.nan)
            for h in range(3):
                for j in range(len(members)):
                    train = fit[mask[fit, h, j]]
                    if len(train) < 100:
                        continue
                    tx, ty = design[train], y[train, h, j]
                    penalty = np.eye(tx.shape[1])*10
                    penalty[0, 0] = 0
                    beta = np.linalg.solve(tx.T@tx+penalty, tx.T@ty)
                    p[:, h, j] = design@beta
                    coefficients.append(dict(task=task, mode=mode, horizon=[5,20,60][h],
                        member=int(members[j]), train_count=len(train), coefficients=beta.tolist()))
            predictions[mode] = p
        for part in ['early', 'calibration', 'validation']:
            ix = parts[part]
            common = mask[ix] & finite[ix, None, None]
            for p in predictions.values():
                common &= np.isfinite(p[ix])
            for h, horizon in enumerate([5, 20, 60]):
                m, truth = common[:, h], y[ix, h]
                n = m.sum(1)
                good = n > 0
                baseline_mse = None
                for mode, p in predictions.items():
                    pred = p[ix, h]
                    daily_mse = np.where(m, (pred-truth)**2, 0).sum(1)/np.maximum(n, 1)
                    mse = float(daily_mse[good].mean()) if good.any() else np.nan
                    if mode == 'fit_entity_mean':
                        baseline_mse = mse
                    if task == 'asset':
                        ic = engine.daily_ic(pred, truth, m)
                        constant = np.array([m[k].sum() >= 3 and np.ptp(pred[k, m[k]]) == 0 for k in range(len(m))])
                        ic[constant] = 0
                        rank_ic = float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan
                    else:
                        rank_ic = np.nan
                    rows.append(dict(part=part, task=task, scope='equity_only', mode=mode,
                        horizon=horizon, dates=int(good.sum()), cells=int(m.sum()), MSE=mse,
                        MSE_ratio_F_entity_mean=mse/baseline_mse if baseline_mse > 1e-16 else np.nan,
                        rank_ic=rank_ic, certified_PIT=False))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'fixed_probe_metrics.csv', index=False)
    summary = table.groupby(['part','task','mode'], sort=False)[['MSE_ratio_F_entity_mean','rank_ic']].mean().reset_index()
    summary.to_csv(OUT / 'fixed_probe_summary.csv', index=False)
    audit = dict(parameter_search=False, ridge_penalty=10, source_definition_certified=False,
        training='F_only_linear_not_itransformer', strict_label_maturity_passed=True,
        split_dates={k:[str(pd.Timestamp(a['dates'][v[0]]).date()), str(pd.Timestamp(a['dates'][v[-1]]).date()), len(v)] for k,v in parts.items()},
        source_split_sha256=sha(split_path), protocol_sha256=sha(ROOT / 'docs/v66_boundary_protocol.md'),
        script_sha256=sha(Path(__file__)), F_scaler_mean=mean.tolist(), F_scaler_std=std.tolist(),
        coefficients=coefficients, promotion=False, portfolio_backtest=False)
    (OUT / 'fixed_probe_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
