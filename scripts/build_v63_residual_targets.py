"""Build forward-only control/residual labels, with an auditable source cutoff."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import v45_forecast as engine
from src.v61_state_momentum import nested_split
from src.v63_residual_learning import historical_control, fit_residual_scale


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = ROOT / 'output/reports/residual_learning_v63'
    data_dir = ROOT / 'data_processed/residual_learning_v63'
    report.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / 'config/v63_residual_learning.json'
    cfg = json.loads(cfg_path.read_text())
    data_path = ROOT / 'data_processed/corporate_action_v58/dataset.npz'
    registry = json.loads((ROOT / 'output/reports/corporate_action_v58/status.json').read_text())
    assert sha(data_path) == registry['dataset_sha256']
    sources = [Path(__file__), ROOT / 'src/v63_residual_learning.py', ROOT / 'src/v61_state_momentum.py',
               ROOT / 'src/v45_forecast.py', ROOT / 'config/v45_research.yaml', cfg_path,
               ROOT / 'docs/v63_residual_learning_protocol.md', data_path]
    identity = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    manifest = report / 'target_source_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Residual target source changed; do not overwrite a frozen experiment')
    manifest.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    with np.load(data_path) as z:
        a = {k: z[k].copy() for k in z.files}
    assert len(a['module_index']) == 38
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    dates, ends = a['dates'], engine.end_times(a)
    coverage, artifacts = [], {}
    for year in cfg['years']:
        split = nested_split(a, engine.shared_split(a, year, old), cfg['calibration_days'], cfg['minimum_inner_train_days'])
        ti = split['train']
        queries = np.concatenate(list(split.values()))
        arrays = {part + '_indices': indices for part, indices in split.items()}
        arrays['dates'] = dates[queries]
        for task, group in groups.items():
            y = a[task + '_raw_relative_target']
            label_mask = (a[task + '_raw_relative_target_mask'].astype(bool)
                & a[task + '_target_mask'].astype(bool) & np.isfinite(y))
            base, count, latest = historical_control(dates, ends, y, label_mask, group, ti, queries,
                history_sessions=cfg['history_sessions'], minimum_group_dates=cfg['minimum_group_dates'],
                minimum_entity_labels=cfg['minimum_entity_labels'], shrinkage=cfg['shrinkage_labels'])
            qualified = label_mask[ti].sum(0) >= old['minimum_train_days']
            residual = y[queries] - base
            masks = label_mask[queries] & np.isfinite(base) & qualified[None]
            scale = fit_residual_scale(residual[:len(ti)], masks[:len(ti)], group)
            masks &= np.isfinite(scale)[None]
            arrays.update({task + '_base': base, task + '_residual': residual,
                task + '_mask': masks, task + '_scale': scale, task + '_source_counts': count,
                task + '_latest_maturity': latest, task + '_qualified': qualified})
            # Confirm removing all non-T labels cannot alter any prediction.
            assert np.all((latest == 0) | (latest < dates[queries, None]))
            future_y = y.copy()
            future_y[np.setdiff1d(np.arange(len(y)), ti)] = 9999.
            check, _, _ = historical_control(dates, ends, future_y, label_mask, group, ti, queries,
                history_sessions=cfg['history_sessions'], minimum_group_dates=cfg['minimum_group_dates'],
                minimum_entity_labels=cfg['minimum_entity_labels'], shrinkage=cfg['shrinkage_labels'])
            np.testing.assert_array_equal(base, check)
            cursor = 0
            for part, indices in split.items():
                m = masks[cursor:cursor+len(indices)]
                for h, horizon in enumerate([5, 20, 60]):
                    coverage.append(dict(year=year, part=part, task=task, horizon=horizon,
                        dates=len(indices), dates_with_any=int(m[:, h].any(1).sum()),
                        entities_with_any=int(m[:, h].any(0).sum()), valid_cells=int(m[:, h].sum())))
                cursor += len(indices)
        path = data_dir / f'{year}.npz'
        if path.exists():
            with np.load(path) as previous:
                assert set(previous.files) == set(arrays)
                for key in arrays:
                    np.testing.assert_array_equal(previous[key], arrays[key])
        else:
            np.savez_compressed(path, **arrays)
        artifacts[str(path.relative_to(ROOT))] = sha(path)
        print(json.dumps(dict(year=year, built=True, labels_mature_before_queries=True)), flush=True)
    pd.DataFrame(coverage).to_csv(report / 'target_coverage.csv', index=False)
    status = dict(ready=True, dataset_sha256=sha(data_path), targets=artifacts,
        source_manifest_sha256=sha(manifest), asset_context_count=38, nine_grid_preserved=True,
        immature_labels_excluded=True, non_T_label_mutation_invariance=True,
        units='cumulative_raw_relative_price_return', formal_promotion=False)
    (report / 'target_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
