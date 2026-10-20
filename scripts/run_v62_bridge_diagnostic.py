"""One fixed-model calibration/coverage diagnostic; no backward pass or selection."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation
from src.cuda_finite_check import noncompacting_finite_check
from src.v60_supervised_adapter import fit_prior
from src.v61_state_momentum import StateRanker, nested_split
from src.v62_centered_views import centered_scores, fit_centered_slopes, comparison_mask
from scripts.run_v61_state_ranker import infer
from scripts.diagnose_v61_completed import daily_rank

OUT = ROOT / 'output/reports/centered_bridge_v62'
NAMES = ['rates', 'credit', 'convertible', 'equity', 'alternatives']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    manifest = json.loads((ROOT / 'output/reports/state_momentum_v61/manifest.json').read_text())
    for name, expected in manifest.items():
        path = ROOT / ('data_processed/corporate_action_v58/dataset.npz' if name == 'dataset_sha256' else name)
        if digest(path) != expected:
            raise ValueError(f'V61 source changed: {name}')


def finite_mean(values):
    values = np.asarray(values)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def metric_rows(year, kind, predictions, prior, truth, eligible, bases, groups):
    rows = []
    for task in groups:
        scopes = [('model_eligible', None)] + [('pair_' + k, k) for k in bases] + [('all_baselines_common', 'all')]
        for scope, baseline in scopes:
            mask = comparison_mask(eligible[task], predictions[task])
            if baseline == 'all':
                for b in bases.values():
                    mask &= np.isfinite(b[task])
            elif baseline:
                mask &= np.isfinite(bases[baseline][task])
            for h, horizon in enumerate([5, 20, 60]):
                for group in np.unique(groups[task]):
                    members = groups[task] == group
                    p, y, m = predictions[task][:, h, members], truth[task][:, h, members], mask[:, h, members]
                    ic = daily_rank(p, y, m)
                    matches = list(bases) if baseline == 'all' else [baseline] if baseline else [None]
                    for matched in matches:
                        base_ic = daily_rank(bases[matched][task][:, h, members], y, m) if matched else np.full(len(m), np.nan)
                        delta = ic - base_ic
                        n = m.sum(1)
                        def mse(v):
                            return finite_mean((np.where(m, v-y, 0)**2).sum(1)[n > 0] / n[n > 0])
                        base_mse = mse(prior[task][:, h, members])
                        ratio = mse(p)/base_mse if kind != 'raw_score' and base_mse > 1e-16 else np.nan
                        rows.append(dict(year=year, kind=kind, task=task, scope=scope, baseline=matched or 'none',
                            group=NAMES[int(group)] if task == 'asset' else 'cross_sleeve', horizon=horizon,
                            valid_cells=int(m.sum()), entities_with_any=int(m.any(0).sum()),
                            valid_rank_dates=int(np.isfinite(ic).sum()), paired_rank_dates=int(np.isfinite(delta).sum()),
                            rank_ic=finite_mean(ic), baseline_rank_ic=finite_mean(base_ic),
                            delta_rank_ic=finite_mean(delta), MSE_ratio_prior=ratio))
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Use the GPU interpreter; do not silently change inference backend')
    verify_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    identity_files = [Path(__file__), ROOT / 'src/v62_centered_views.py', ROOT / 'docs/v62_centered_bridge_diagnostic_protocol.md',
                      ROOT / 'scripts/diagnose_v61_completed.py']
    cfg = json.loads((ROOT / 'config/v61_state_momentum.json').read_text())
    for year in cfg['years']:
        for seed in cfg['seeds']:
            folder = ROOT / f'output/models/state_momentum_v61/{year}/{seed}'
            identity_files += [folder / 'selected.pt', folder / 'validation_predictions.npz']
    identity = {str(p.relative_to(ROOT)): digest(p) for p in identity_files}
    identity_path = OUT / 'source_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Diagnostic identity changed; retain previous results and use a new version')
    identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    with np.load(ROOT / 'data_processed/corporate_action_v58/dataset.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / 'data_processed/state_momentum_v61/signals.npz') as z:
        signals = {k: z[k].copy() for k in z.files}
    extras = np.stack([2*signals['cohort']-1, np.tanh(signals['extension']/3), signals['value_boundary']], -1)
    a['local_values'] = np.concatenate([a['local_values'], np.nan_to_num(extras)], -1).astype(np.float32)
    a['local_masks'] = np.concatenate([a['local_masks'], np.isfinite(extras)], -1)
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda')
    all_rows, slope_rows, support_rows = [], [], []
    with training_implementation(), noncompacting_finite_check(), torch.no_grad():
        for year in cfg['years']:
            split = nested_split(a, engine.shared_split(a, year, old), cfg['calibration_days'], cfg['minimum_inner_train_days'])
            ti, ci, vi = (split[k] for k in ['train', 'calibration', 'validation'])
            clipped, _ = engine.clip_fold_features(a, ti, old)
            prior, maturity_mask, asof = {}, {}, {}
            for task in groups:
                p, _, _, available, count = fit_prior(a[task + '_raw_relative_target'][ti], a[task + '_target_mask'][ti], groups[task])
                prior[task] = p
                qualified = available & (count >= old['minimum_train_days'])
                visible = a['local_masks'].any(-1) if task == 'asset' else np.stack([
                    a['local_masks'][:, a['module_index'] == g].any((1, 2)) for g in range(5)], -1)
                asof[task] = visible[:, None, :] & qualified[None]
                maturity_mask[task] = a[task + '_target_mask'].astype(bool) & a[task + '_raw_relative_target_mask'].astype(bool) & np.isfinite(a[task + '_raw_relative_target']) & asof[task]
            truth = {t: a[t + '_raw_relative_target'][vi] for t in groups}
            priors = {t: np.broadcast_to(prior[t], truth[t].shape) for t in groups}
            bases = {'prior': priors}
            for name in ['momentum20', 'cohort', 'price_soft']:
                x, w = signals[name], a['sleeve_reference_weights']
                sleeve = np.sum(w*np.where(np.isfinite(x[:, None]), x[:, None], 0), -1)
                good = (w.sum(-1) > 0) & ((w == 0) | np.isfinite(x[:, None])).all(-1)
                bases[name] = dict(asset=np.repeat(x[:, None], 3, 1)[vi], sleeve=np.repeat(np.where(good, sleeve, np.nan)[:, None], 3, 1)[vi])
            saved_scores, old_means, new_means = [], [], []
            for seed in cfg['seeds']:
                folder = ROOT / f'output/models/state_momentum_v61/{year}/{seed}'
                selected = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
                model = StateRanker(engine.ForecastNetwork(clipped, old), old['d_model']).to(device)
                model.load_state_dict(selected['state'])
                cv = infer(model, clipped, np.concatenate([ci, vi]), old, device, cfg['batch_size'])
                with np.load(folder / 'validation_predictions.npz') as z:
                    stored = {k: z[k].copy() for k in z.files}
                means = {}
                for task in groups:
                    cs, vs = cv[task][:len(ci)], cv[task][len(ci):]
                    np.testing.assert_allclose(vs, stored[task], rtol=1e-4, atol=1e-5)
                    cx = centered_scores(cs, asof[task][ci], groups[task])
                    vx = centered_scores(vs, asof[task][vi], groups[task])
                    beta, details = fit_centered_slopes(cx, a[task + '_raw_relative_target'][ci], maturity_mask[task][ci], prior[task], groups[task])
                    means[task] = prior[task] + vx * beta
                    slope_rows.extend(dict(year=year, seed=seed, task=task, **row) for row in details)
                saved_scores.append({t: stored[t] for t in groups})
                old_means.append({t: stored[t + '_mean'] for t in groups})
                new_means.append(means)
                print(json.dumps(dict(year=year, seed=seed, inference_parity=True, complete=True)), flush=True)
                del model
                torch.cuda.empty_cache()
            eligible = {t: maturity_mask[t][vi] for t in groups}
            for kind, values in [('raw_score', saved_scores), ('old_mean', old_means), ('centered_mean', new_means)]:
                pred = {t: np.mean([v[t] for v in values], 0) for t in groups}
                all_rows.extend(metric_rows(year, kind, pred, priors, truth, eligible, bases, groups))
            for task in groups:
                universe = len(groups[task])
                for scope in ['model_eligible'] + ['pair_' + b for b in bases] + ['all_baselines_common']:
                    mask = eligible[task].copy()
                    for b in bases:
                        if scope == 'all_baselines_common' or scope == 'pair_' + b:
                            mask &= np.isfinite(bases[b][task])
                    support_rows.append(dict(year=year, task=task, scope=scope, universe=universe,
                        entities_with_any=int(mask.any((0, 1)).sum()), valid_cells=int(mask.sum())))
    verify_sources()
    for name, expected in identity.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Artifact changed during diagnostic')
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / 'component_metrics.csv', index=False)
    pd.DataFrame(slope_rows).to_csv(OUT / 'calibration_slopes.csv', index=False)
    pd.DataFrame(support_rows).to_csv(OUT / 'coverage.csv', index=False)
    metrics = ['rank_ic', 'baseline_rank_ic', 'delta_rank_ic', 'MSE_ratio_prior']
    task_summary = df.groupby(['year', 'kind', 'scope', 'baseline', 'task'], as_index=False)[metrics].mean()
    summary = task_summary.groupby(['year', 'kind', 'scope', 'baseline'], as_index=False)[metrics].mean()
    task_summary.to_csv(OUT / 'task_summary.csv', index=False)
    summary.to_csv(OUT / 'summary.csv', index=False)
    status = dict(completed=True, models_inferred=6, trained=False, original_V_predictions_reproduced=True,
        source_hashes_verified=True, exploratory_V_only=True, G_or_test_used=False, formal_promotion=False,
        ppo_training_started=False, next_step='stop; review coverage and calibration, no automatic search')
    (OUT / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
