"""Rebuild F-only history and scales; do not reuse E-exposed V63 fitting."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import v45_forecast as engine
from src.v64_task_ablation import split_inner
from src.v63_residual_learning import historical_control, fit_residual_scale
from scripts.build_v63_residual_targets import sha


def main():
    out = ROOT / 'output/reports/task_ablation_v64'
    target = ROOT / 'data_processed/task_ablation_v64'
    out.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'config/v64_task_ablation.json').read_text())
    parent = json.loads((ROOT / cfg['parent_config']).read_text())
    files = [Path(__file__), ROOT / 'src/v64_task_ablation.py', ROOT / 'src/v63_residual_learning.py',
        ROOT / 'src/v45_forecast.py', ROOT / 'scripts/build_v63_residual_targets.py',
        ROOT / 'config/v64_task_ablation.json', ROOT / cfg['parent_config'], ROOT / 'config/v45_research.yaml',
        ROOT / 'docs/v64_task_ablation_protocol.md', ROOT / 'data_processed/corporate_action_v58/dataset.npz']
    for y in cfg['candidate_years']:
        files.append(ROOT / f'data_processed/residual_learning_v63/{y}.npz')
    identity = {str(p.relative_to(ROOT)): sha(p) for p in files}
    manifest = out / 'target_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Target definition changed; incompatible overwrite refused')
    manifest.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    with np.load(ROOT / 'data_processed/corporate_action_v58/dataset.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    status_v63 = json.loads((ROOT / 'output/reports/residual_learning_v63/target_status.json').read_text())
    assert sha(ROOT / 'data_processed/corporate_action_v58/dataset.npz') == status_v63['dataset_sha256']
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    ends, dates = engine.end_times(a), a['dates']
    artifacts, splits, coverage = {}, [], []
    for year in cfg['candidate_years']:
        source = ROOT / f'data_processed/residual_learning_v63/{year}.npz'
        assert sha(source) == status_v63['targets'][str(source.relative_to(ROOT))]
        with np.load(source) as z:
            original = {k: z[k].copy() for k in z.files}
        inner = split_inner(original['train_indices'], ends, dates, cfg['inner_validation_days'], cfg['minimum_fit_days'])
        if inner is None:
            splits.append(dict(year=year, eligible=False, reason='insufficient_F_after_E_label_purge'))
            continue
        split = dict(**inner, calibration=original['calibration_indices'], validation=original['validation_indices'])
        for earlier, later in [('fit', 'early'), ('early', 'calibration'), ('calibration', 'validation')]:
            assert (ends[split[earlier]].max(1) < dates[split[later][0]]).all()
        fit = split['fit']
        query = np.concatenate(list(split.values()))
        arrays = {k+'_indices': v for k, v in split.items()}
        for task, g in groups.items():
            y = a[task+'_raw_relative_target']
            label_mask = a[task+'_target_mask'].astype(bool) & a[task+'_raw_relative_target_mask'].astype(bool) & np.isfinite(y)
            kwargs = dict(history_sessions=parent['history_sessions'], minimum_group_dates=parent['minimum_group_dates'],
                          minimum_entity_labels=parent['minimum_entity_labels'], shrinkage=parent['shrinkage_labels'])
            base, _, latest = historical_control(dates, ends, y, label_mask, g, fit, query, **kwargs)
            qualified = label_mask[fit].sum(0) >= cfg['minimum_fit_days']
            residual = y[query]-base
            masks = label_mask[query] & np.isfinite(base) & qualified[None]
            sigma = fit_residual_scale(residual[:len(fit)], masks[:len(fit)], g)
            if not np.isfinite(sigma).all():
                raise ValueError('Insufficient F scale support')
            modified = y.copy()
            modified[np.setdiff1d(np.arange(len(y)), fit)] = 9999.
            test, _, _ = historical_control(dates, ends, modified, label_mask, g, fit, query, **kwargs)
            np.testing.assert_array_equal(base, test)
            arrays.update({task+'_base': base, task+'_residual': residual, task+'_mask': masks,
                task+'_scale': sigma, task+'_latest_maturity': latest})
            cursor = 0
            for part, ix in split.items():
                m = masks[cursor:cursor+len(ix)]
                for h, horizon in enumerate([5, 20, 60]):
                    coverage.append(dict(year=year, part=part, task=task, horizon=horizon,
                        dates=len(ix), valid_dates=int(m[:, h].any(1).sum()),
                        valid_cells=int(m[:, h].sum()), entities_with_any=int(m[:, h].any(0).sum())))
                cursor += len(ix)
        path = target / f'{year}.npz'
        if path.exists():
            with np.load(path) as z:
                assert set(z.files) == set(arrays)
                for k in arrays:
                    np.testing.assert_array_equal(z[k], arrays[k])
        else:
            np.savez_compressed(path, **arrays)
        artifacts[str(path.relative_to(ROOT))] = sha(path)
        splits.append(dict(year=year, eligible=True, parts={k: dict(n=len(v), start=str(pd.Timestamp(dates[v[0]]).date()), end=str(pd.Timestamp(dates[v[-1]]).date())) for k, v in split.items()}))
    pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
    result = dict(ready=True, splits=splits, artifacts=artifacts, tasks=9,
        sources_fit_only=True, non_F_label_mutation_invariance=True,
        formal_promotion=False, target_manifest_sha256=sha(manifest))
    (out / 'target_status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
