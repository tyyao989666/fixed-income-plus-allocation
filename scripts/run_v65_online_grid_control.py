"""Fixed online-control replay, grid support audit and future-model target contract."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import v45_forecast as engine
from src.v45_data import observable_grid, GRID_NAMES
from src.v63_residual_learning import fit_residual_scale
from src.v65_online_grid_control import online_grid_control
from scripts.build_v63_residual_targets import sha
from scripts.diagnose_v61_completed import daily_rank

OUT = ROOT / 'output/reports/online_grid_control_v65'


def evaluate(a, r, query, parts, predictions):
    rows = []
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    group_names = ['rates', 'credit', 'convertible', 'equity', 'alternatives']
    for part, sl in parts.items():
        for task, group in groups.items():
            truth = a[task+'_raw_relative_target'][query[sl]]
            mask = r[task+'_mask'][sl].astype(bool)
            for values in predictions.values():
                mask &= np.isfinite(values[task][sl])
            frozen = predictions['frozen_mean'][task][sl]
            for h, horizon in enumerate([5, 20, 60]):
                for g in np.unique(group):
                    members = group == g
                    m, y, b = mask[:, h, members], truth[:, h, members], frozen[:, h, members]
                    n = m.sum(1)
                    def mse(p, sub):
                        good = n[sub] > 0
                        if not good.any():
                            return np.nan
                        error = (np.where(m[sub], p[sub]-y[sub], 0)**2).sum(1)/np.maximum(n[sub], 1)
                        return float(error[good].mean())
                    for name, values in predictions.items():
                        p = values[task][sl][:, h, members]
                        for block, sub in [('all', slice(None))] + ([(f'block_{i//60+1}', slice(i, min(i+60, len(m)))) for i in range(0, len(m), 60)] if part == 'early' else []):
                            bmse = mse(b, sub)
                            pmse = mse(p, sub)
                            ic, bic = daily_rank(p[sub], y[sub], m[sub]), daily_rank(b[sub], y[sub], m[sub])
                            rows.append(dict(part=part, task=task, group=group_names[int(g)] if task == 'asset' else 'cross_sleeve',
                                horizon=horizon, control=name, block=block, period_dates=len(n[sub]),
                                valid_cells=int(m[sub].sum()), matched_rank_dates=int(np.isfinite(ic-bic).sum()),
                                MSE_ratio_frozen=pmse/bmse if bmse > 1e-16 else np.nan,
                                rank_ic=engine.finite_mean(ic), frozen_rank_ic=engine.finite_mean(bic),
                                delta_rank_ic=engine.finite_mean(ic-bic)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'config/v65_online_grid_control.json').read_text())
    data_path, split_path = ROOT / cfg['source_dataset'], ROOT / cfg['source_split']
    status_v64 = json.loads((ROOT / 'output/reports/task_ablation_v64/target_status.json').read_text())
    assert sha(split_path) == status_v64['artifacts'][str(split_path.relative_to(ROOT))]
    parent_status = json.loads((ROOT / 'output/reports/residual_learning_v63/target_status.json').read_text())
    assert sha(data_path) == parent_status['dataset_sha256']
    files = [Path(__file__), ROOT / 'src/v65_online_grid_control.py', ROOT / 'src/v45_data.py',
        ROOT / 'src/v45_forecast.py', ROOT / 'src/v63_residual_learning.py', ROOT / 'scripts/diagnose_v61_completed.py',
        ROOT / 'config/v65_online_grid_control.json', ROOT / 'docs/v65_online_grid_control_protocol.md',
        ROOT / 'output/reports/corporate_action_v58/metadata.json', data_path, split_path]
    identity = {str(p.relative_to(ROOT)): sha(p) for p in files}
    frozen = OUT / 'source_manifest.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('V65 source identity changed; previous experiment must be preserved')
    frozen.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    with np.load(data_path) as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(split_path) as z:
        r = {k: z[k].copy() for k in z.files}
    meta = json.loads((ROOT / 'output/reports/corporate_action_v58/metadata.json').read_text(encoding='utf-8'))
    assert len(meta['assets']) == len(a['module_index']) == 38
    returns = pd.DataFrame(np.where(a['return_masks'], a['returns'].astype(np.float64), np.nan),
        index=pd.to_datetime(a['dates']), columns=meta['assets'])
    macro, macro_masks, regimes = observable_grid(returns)
    np.testing.assert_array_equal(regimes, a['regime_index'])
    # Returns were serialized as float32 after original feature construction.
    # Discrete states/masks must remain exact; score round trips need not be bitwise.
    np.testing.assert_array_equal(macro[:, :9], a['macro_values'][:, :9])
    score_error = float(np.max(np.abs(macro[:, 9:]-a['macro_values'][:, 9:])))
    score_tolerance = 20*np.finfo(a['returns'].dtype).eps
    np.testing.assert_allclose(macro[:, 9:], a['macro_values'][:, 9:], rtol=0, atol=score_tolerance)
    np.testing.assert_array_equal(macro_masks, a['macro_masks'])
    parts, cursor, queries = {}, 0, []
    for part in ['fit', 'early', 'calibration', 'validation']:
        ix = r[part+'_indices']
        parts[part] = slice(cursor, cursor+len(ix))
        queries.append(ix)
        cursor += len(ix)
    query = np.concatenate(queries)
    sources = np.arange(r['fit_indices'][0], len(a['dates']))
    ends, dates = engine.end_times(a), a['dates']
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    predictions = {name: {} for name in cfg['controls']}
    kwargs = dict(history_sessions=cfg['history_sessions'], half_life_sessions=cfg['half_life_sessions'],
        minimum_group_dates=cfg['minimum_group_dates'], shrinkage=cfg['shrinkage_effective_labels'])
    support, episode_rows, audits, export = [], [], [], {'query_indices': query, 'dates': dates[query], 'regime_index': regimes[query]}
    for part, ix in [('full_dataset', np.arange(len(dates)))] + [(k, r[k+'_indices']) for k in parts]:
        local_states = regimes[ix]
        for grid in range(-1, 9):
            selected = local_states == grid
            starts = selected & np.r_[True, (~selected[:-1]) | (np.diff(ix) != 1)]
            episode_rows.append(dict(part=part, grid=grid, state='unknown' if grid < 0 else GRID_NAMES[grid],
                days=int(selected.sum()), contiguous_runs=int(starts.sum()), proportion=float(selected.mean())))
    for task, g in groups.items():
        y = a[task+'_raw_relative_target']
        mask = a[task+'_target_mask'].astype(bool) & a[task+'_raw_relative_target_mask'].astype(bool) & np.isfinite(y)
        online = online_grid_control(dates, ends, y, mask, g, regimes, sources, query, **kwargs)
        predictions['frozen_mean'][task] = r[task+'_base']
        for name in ['online_mean', 'online_grid']:
            predictions[name][task] = online[name]
            residual = y[query]-online[name]
            valid = r[task+'_mask'].astype(bool) & np.isfinite(online[name])
            sigma = fit_residual_scale(residual[parts['fit']], valid[parts['fit']], g)
            export[task+'_'+name+'_base'] = online[name]
            export[task+'_'+name+'_residual'] = residual
            export[task+'_'+name+'_mask'] = valid
            export[task+'_'+name+'_F_scale'] = sigma
        for key in ['latest_maturity', 'joint_weight_ESS', 'history_counts', 'mean_information_age_sessions']:
            export[task+'_'+key] = online[key]
        # Real-data causal replay: future and same-day labels/states are unusable.
        test_query = int(r['early_indices'][-1])
        modified = y.copy()
        for h in range(3):
            modified[(ends[:, h] >= dates[test_query]) | (np.arange(len(dates)) >= test_query), h] = 9999.
        before = online_grid_control(dates, ends, y, mask, g, regimes, sources, [test_query], **kwargs)
        altered = online_grid_control(dates, ends, modified, mask, g, regimes, sources, [test_query], **kwargs)
        prefix = online_grid_control(dates[:test_query+1], ends[:test_query+1], y[:test_query+1], mask[:test_query+1], g,
            regimes[:test_query+1], sources[sources <= test_query], [test_query], **kwargs)
        for name in ['online_mean', 'online_grid']:
            np.testing.assert_array_equal(before[name], altered[name])
            np.testing.assert_array_equal(before[name], prefix[name])
        assert np.all((online['latest_maturity'] == 0) | (online['latest_maturity'] < dates[query, None]))
        for part, sl in parts.items():
            for grid in range(-1, 9):
                choose = regimes[query[sl]] == grid
                for h, horizon in enumerate([5, 20, 60]):
                    m = r[task+'_mask'][sl][:, h] & np.isfinite(online['online_grid'][sl][:, h])
                    m &= choose[:, None]
                    ess = online['joint_weight_ESS'][sl][:, h][m]
                    support.append(dict(part=part, task=task, grid=grid, horizon=horizon,
                        state='unknown' if grid < 0 else GRID_NAMES[grid], state_days=int(choose.sum()),
                        valid_label_cells=int(m.sum()), entities_with_any=int(m.any(0).sum()),
                        mean_joint_weight_ESS=float(ess.mean()) if len(ess) else np.nan,
                        mean_joint_blend_weight=float((ess/(ess+cfg['shrinkage_effective_labels'])).mean()) if len(ess) else np.nan))
        audits.append(dict(task=task, latest_label_strictly_past=True, future_label_mutation_invariance=True,
            prefix_replay_equal=True, context_entities=len(g)))
        print(json.dumps(dict(task=task, replay_complete=True)), flush=True)
    metrics = evaluate(a, r, query, parts, predictions)
    metrics.to_csv(OUT / 'control_component_metrics.csv', index=False)
    task_summary = metrics[metrics.block == 'all'].groupby(['part', 'control', 'task'], as_index=False)[
        ['MSE_ratio_frozen', 'rank_ic', 'frozen_rank_ic', 'delta_rank_ic']].mean()
    task_summary.to_csv(OUT / 'task_summary.csv', index=False)
    task_summary.groupby(['part', 'control'], as_index=False)[['MSE_ratio_frozen', 'rank_ic', 'frozen_rank_ic', 'delta_rank_ic']].mean().to_csv(OUT / 'overall_summary.csv', index=False)
    pd.DataFrame(episode_rows).to_csv(OUT / 'grid_day_and_episode_counts.csv', index=False)
    pd.DataFrame(support).to_csv(OUT / 'grid_label_support.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'causality_audit.csv', index=False)
    dest = ROOT / 'data_processed/online_grid_control_v65'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'research_targets.npz', **export)
    for name, expected in identity.items():
        assert sha(ROOT / name) == expected
    result = dict(completed=True, source_hashes_unchanged=True, grid_states_and_masks_reconstruction_exact=True,
        continuous_grid_scores_max_error=score_error, continuous_grid_scores_tolerance=float(score_tolerance),
        continuous_grid_scores_bitwise_exact=False,
        real_data_prefix_and_future_mutation_passed=True, online_labels_matured_strictly_before_query=True,
        context_assets=38, neural_training_started=False, formal_promotion=False, ppo_training_started=False,
        parameters_selected_on_V=False, research_only=True,
        target_path=str((dest / 'research_targets.npz').relative_to(ROOT)), target_sha256=sha(dest / 'research_targets.npz'))
    (OUT / 'status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
