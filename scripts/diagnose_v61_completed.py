"""Read-only model diagnosis; writes separate tables, never trains or selects models."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v60_supervised_adapter import fit_prior
from src.v61_state_momentum import nested_split
from scripts.run_v59_raw_mean import sha
from scripts.run_v61_state_ranker import rank_metrics


def daily_rank(p, y, m):
    result = engine.daily_ic(p, y, m)
    for i in range(len(result)):
        good = m[i] & np.isfinite(p[i]) & np.isfinite(y[i])
        if good.sum() >= 2 and np.ptp(y[i, good]) > 0 and np.ptp(p[i, good]) == 0:
            result[i] = 0.
    return result


def main():
    out = ROOT / 'output/reports/state_momentum_v61'
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, expected in manifest.items():
        path = ROOT / ('data_processed/corporate_action_v58/dataset.npz' if name == 'dataset_sha256' else name)
        if sha(path) != expected:
            raise ValueError(f'Source identity changed: {name}')
    cfg = json.loads((ROOT / 'config/v61_state_momentum.json').read_text())
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    with np.load(ROOT / 'data_processed/corporate_action_v58/dataset.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / 'data_processed/state_momentum_v61/signals.npz') as z:
        signals = {k: z[k].copy() for k in z.files}
    metadata = json.loads((ROOT / 'output/reports/corporate_action_v58/metadata.json').read_text(encoding='utf-8'))
    asset_ids = metadata['assets']
    assert len(asset_ids) == len(a['module_index']) == 38
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    names = ['rates', 'credit', 'convertible', 'equity', 'alternatives']
    detail, calibration, coverage, bridge = [], [], [], []
    for year in cfg['years']:
        split = nested_split(a, engine.shared_split(a, year, old), cfg['calibration_days'], cfg['minimum_inner_train_days'])
        ti, vi = split['train'], split['validation']
        truth, masks, priors, train_counts = {}, {}, {}, {}
        for task in groups:
            truth[task] = a[task + '_raw_relative_target'][vi]
            p, _, _, available, count = fit_prior(a[task + '_raw_relative_target'][ti], a[task + '_target_mask'][ti], groups[task])
            priors[task] = p
            train_counts[task] = count
            masks[task] = a[task + '_target_mask'][vi].astype(bool) & available[None] & (count[None] >= old['minimum_train_days'])
        bases = {'prior': {t: np.broadcast_to(priors[t], truth[t].shape) for t in groups}}
        for name in ['momentum20', 'cohort', 'price_soft']:
            x, weights = signals[name], a['sleeve_reference_weights']
            sleeve = np.sum(weights * np.where(np.isfinite(x[:, None]), x[:, None], 0), -1)
            valid = (weights.sum(-1) > 0) & ((weights == 0) | np.isfinite(x[:, None])).all(-1)
            bases[name] = dict(asset=np.repeat(x[:, None], 3, 1)[vi], sleeve=np.repeat(np.where(valid, sleeve, np.nan)[:, None], 3, 1)[vi])
        eligible_masks = {t: m.copy() for t, m in masks.items()}
        for base in bases.values():
            for task in groups:
                masks[task] &= np.isfinite(base[task])
        seed_predictions = []
        for seed in cfg['seeds']:
            folder = ROOT / f'output/models/state_momentum_v61/{year}/{seed}'
            with np.load(folder / 'validation_predictions.npz') as z:
                values = {k: z[k].copy() for k in z.files}
            checkpoint = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
            for task in groups:
                reconstructed = checkpoint['prior'][task] + checkpoint['calibrator'][task] * values[task]
                np.testing.assert_allclose(reconstructed, values[task + '_mean'], rtol=1e-6, atol=1e-8)
            seed_predictions.append(values)
            for task in groups:
                for h, horizon in enumerate(a['horizons_days']):
                    for group in np.unique(groups[task]):
                        members = groups[task] == group
                        beta = checkpoint['calibrator'][task][h, members]
                        m = masks[task][:, h, members]
                        calibration.append(dict(year=year, seed=seed, task=task, group=names[int(group)] if task == 'asset' else 'cross_sleeve', horizon=int(horizon), slope=float(beta[0]), positive=bool(beta[0] > 0), valid_cells=int(m.sum())))
        ensemble = {key: np.mean([p[key] for p in seed_predictions], 0) for key in seed_predictions[0]}
        for label, values in [(str(s), p) for s, p in zip(cfg['seeds'], seed_predictions)] + [('ensemble', ensemble)]:
            for kind in ['rank_score', 'calibrated_mean']:
                pred = {t: values[t if kind == 'rank_score' else t + '_mean'] for t in groups}
                minimum, deltas, ic = rank_metrics(pred, bases, truth, masks, groups)
                bridge.append(dict(year=year, model=label, kind=kind, rank_ic=ic, minimum_increment=minimum, **{'delta_' + k: v for k, v in deltas.items()}))
        for task in groups:
            for entity in range(len(groups[task])):
                m = masks[task][:, :, entity]
                coverage.append(dict(year=year, task=task, entity_index=entity,
                    entity_id=asset_ids[entity] if task == 'asset' else names[entity],
                    group=names[int(groups[task][entity])] if task == 'asset' else 'cross_sleeve',
                    minimum_train_count=int(train_counts[task][:, entity].min()),
                    raw_label_cells=int(a[task + '_target_mask'][vi, :, entity].sum()),
                    train_eligible_validation_cells=int(eligible_masks[task][:, :, entity].sum()),
                    **{k + '_finite_cells': int(np.isfinite(v[task][:, :, entity]).sum()) for k, v in bases.items()},
                    valid_cells=int(m.sum()), dates_with_any=int(m.any(1).sum()), total_dates=len(vi)))
            for h, horizon in enumerate(a['horizons_days']):
                for group in np.unique(groups[task]):
                    members = groups[task] == group
                    m, y = masks[task][:, h, members], truth[task][:, h, members]
                    base_ic = {k: daily_rank(v[task][:, h, members], y, m) for k, v in bases.items()}
                    for kind, key in [('rank_score', task), ('calibrated_mean', task + '_mean')]:
                        p = ensemble[key][:, h, members]
                        ic = daily_rank(p, y, m)
                        n = m.sum(1)
                        mse = lambda v: float(np.mean((np.where(m, v-y, 0)**2).sum(1)[n > 0] / n[n > 0])) if (n > 0).any() else np.nan
                        for baseline, b in bases.items():
                            delta = ic - base_ic[baseline]
                            # Nonoverlapping 60-session chunks are descriptive, not independent-test p-values.
                            chunks = [engine.finite_mean(delta[start:start+60]) for start in range(0, len(delta), 60)]
                            detail.append(dict(year=year, task=task, group=names[int(group)] if task == 'asset' else 'cross_sleeve', horizon=int(horizon), kind=kind, baseline=baseline,
                                valid_rank_dates=int(np.isfinite(delta).sum()), rank_ic=engine.finite_mean(ic), baseline_rank_ic=engine.finite_mean(base_ic[baseline]), delta_rank_ic=engine.finite_mean(delta),
                                MSE_ratio_prior=mse(p)/mse(bases['prior'][task][:, h, members]),
                                chunk_1=chunks[0], chunk_2=chunks[1], chunk_3=chunks[2]))
    for name, rows in [('component_diagnosis', detail), ('calibration_slopes', calibration), ('entity_validation_coverage', coverage), ('score_to_mean_bridge', bridge)]:
        pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False)
    decision = json.loads((out / 'decision.json').read_text())
    for entry in decision['details']:
        row = next(r for r in bridge if r['year'] == entry['year'] and r['model'] == 'ensemble' and r['kind'] == 'rank_score')
        np.testing.assert_allclose(row['rank_ic'], entry['rank_ic'], atol=1e-10)
        np.testing.assert_allclose(row['minimum_increment'], entry['min_rank_increment'], atol=1e-10)
    result = dict(training_started=False, model_selection_changed=False, G_or_test_used=False,
        stored_means_reconstructed=True, original_decision_metrics_reproduced=True,
        formal_promotion=False, diagnosis_script_sha256=sha(Path(__file__)),
        positive_calibrators=int(sum(r['positive'] for r in calibration)), total_calibrators=len(calibration),
        ensemble_bridge=[r for r in bridge if r['model'] == 'ensemble'])
    (out / 'completed_diagnosis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
