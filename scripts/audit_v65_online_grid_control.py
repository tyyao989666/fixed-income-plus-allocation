"""Verify exact raw grid lineage and matched-control replay accounting."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v45_data import observable_grid
from scripts.build_v63_residual_targets import sha


def main():
    out = ROOT / 'output/reports/online_grid_control_v65'
    manifest = json.loads((out / 'source_manifest.json').read_text())
    for name, expected in manifest.items():
        assert sha(ROOT / name) == expected, name
    status = json.loads((out / 'status.json').read_text())
    assert status['completed'] and not status['neural_training_started'] and not status['ppo_training_started']
    assert sha(ROOT / status['target_path']) == status['target_sha256']
    cfg = json.loads((ROOT / 'config/v65_online_grid_control.json').read_text())
    with np.load(ROOT / cfg['source_dataset']) as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / cfg['source_split']) as z:
        r = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / status['target_path']) as z:
        exported = {k: z[k].copy() for k in z.files}
    meta = json.loads((ROOT / 'output/reports/corporate_action_v58/metadata.json').read_text(encoding='utf-8'))
    # Use only the two unaffected grid axes, never re-ingest retired credit returns.
    path = ROOT / 'data_processed/unified_v53/research_returns_daily_38.parquet'
    raw_hash = sha(path)
    axes = ['rate_7_10y', 'equity_broad']
    raw = pd.read_parquet(path, columns=axes)
    np.testing.assert_array_equal(raw.index.view('int64'), a['dates'])
    for name in axes:
        j = meta['assets'].index(name)
        values = raw[name].to_numpy()
        np.testing.assert_array_equal(np.isfinite(values), a['return_masks'][:, j].astype(bool))
        np.testing.assert_array_equal(values[np.isfinite(values)].astype(np.float32), a['returns'][:, j][a['return_masks'][:, j].astype(bool)])
    macro, masks, regimes = observable_grid(raw)
    np.testing.assert_array_equal(macro, a['macro_values'])
    np.testing.assert_array_equal(masks, a['macro_masks'])
    np.testing.assert_array_equal(regimes, a['regime_index'])
    frame = pd.read_csv(out / 'control_component_metrics.csv')
    keys = ['part', 'task', 'group', 'horizon', 'block']
    for _, block in frame.groupby(keys):
        assert set(block.control) == set(cfg['controls'])
        assert block.valid_cells.nunique() == 1
        # All three controls are evaluated on the same finite support.
        finite = block.dropna(subset=['delta_rank_ic'])
        np.testing.assert_allclose(finite.rank_ic-finite.frozen_rank_ic, finite.delta_rank_ic, atol=1e-12)
        baseline = block[block.control == 'frozen_mean'].iloc[0]
        if np.isfinite(baseline.MSE_ratio_frozen):
            np.testing.assert_allclose(baseline.MSE_ratio_frozen, 1., atol=1e-12)
    query = exported['query_indices']
    np.testing.assert_array_equal(exported['dates'], a['dates'][query])
    total = 0
    for part in ['fit', 'early', 'calibration', 'validation']:
        expected = r[part+'_indices']
        np.testing.assert_array_equal(query[total:total+len(expected)], expected)
        total += len(expected)
    for task in ['asset', 'sleeve']:
        latest = exported[task+'_latest_maturity']
        assert np.all((latest == 0) | (latest < a['dates'][query, None]))
        for control in ['online_mean', 'online_grid']:
            base = exported[task+'_'+control+'_base']
            residual = exported[task+'_'+control+'_residual']
            valid = exported[task+'_'+control+'_mask']
            np.testing.assert_allclose(residual[valid], (a[task+'_raw_relative_target'][query]-base)[valid], atol=1e-10)
            assert np.isfinite(base[valid]).all()
            sigma = exported[task+'_'+control+'_F_scale']
            assert np.isfinite(sigma).all() and (sigma > 0).all()
    assert sha(path) == raw_hash
    for name, expected in manifest.items():
        assert sha(ROOT / name) == expected
    result = dict(passed=True, exact_grid_reconstructed_from_original_precision=True,
        discrete_and_continuous_grid_exact=True, retired_credit_returns_not_loaded=True,
        grid_axes_source=str(path.relative_to(ROOT)), grid_axes_source_sha256=raw_hash,
        grid_axes_columns_read=axes, matched_comparison_masks=True,
        residual_contract_reproduced=True, latest_labels_strictly_past=True,
        original_source_hashes_unchanged=True, no_neural_training=True,
        formal_promotion=False, audit_script_sha256=sha(Path(__file__)))
    (out / 'artifact_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
