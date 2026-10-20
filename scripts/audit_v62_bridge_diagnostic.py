"""Verify completed fixed-model diagnostic accounting and pairwise comparability."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v62_bridge_diagnostic import digest, verify_sources


def main():
    out = ROOT / 'output/reports/centered_bridge_v62'
    verify_sources()
    identity = json.loads((out / 'source_identity.json').read_text())
    for name, expected in identity.items():
        assert digest(ROOT / name) == expected, name
    status = json.loads((out / 'status.json').read_text())
    assert status['completed'] and status['models_inferred'] == 6
    assert status['original_V_predictions_reproduced']
    assert not any(status[k] for k in ['trained', 'G_or_test_used', 'formal_promotion', 'ppo_training_started'])
    d = pd.read_csv(out / 'component_metrics.csv')
    c = pd.read_csv(out / 'coverage.csv')
    slopes = pd.read_csv(out / 'calibration_slopes.csv')
    assert len(slopes) == 108
    assert np.isfinite(slopes.slope).all() and (slopes.slope >= 0).all()
    assert ((slopes.slope > 0) == (slopes.reason == 'positive')).all()
    assert d[d.kind == 'raw_score'].MSE_ratio_prior.isna().all()
    rank = d.dropna(subset=['delta_rank_ic'])
    np.testing.assert_allclose(rank.delta_rank_ic, rank.rank_ic-rank.baseline_rank_ic, atol=1e-12)
    keys = ['year', 'task', 'scope', 'baseline', 'group', 'horizon']
    for _, block in d.groupby(keys):
        assert len(block) == 3 and block.kind.nunique() == 3
        assert block.valid_cells.nunique() == 1
        assert block.entities_with_any.nunique() == 1
        assert block.valid_rank_dates.nunique() == 1
        assert block.baseline_rank_ic.dropna().nunique() <= 1
    for (_, _), block in c.groupby(['year', 'task']):
        indexed = block.set_index('scope')
        full = indexed.loc['model_eligible']
        common = indexed.loc['all_baselines_common']
        for _, row in block.iterrows():
            assert common.valid_cells <= row.valid_cells <= full.valid_cells
            assert common.entities_with_any <= row.entities_with_any <= full.entities_with_any <= row.universe
    summary = pd.read_csv(out / 'summary.csv')
    tasks = pd.read_csv(out / 'task_summary.csv')
    rebuilt = tasks.groupby(['year', 'kind', 'scope', 'baseline'], as_index=False)[
        ['rank_ic', 'baseline_rank_ic', 'delta_rank_ic', 'MSE_ratio_prior']].mean()
    pd.testing.assert_frame_equal(summary, rebuilt, check_exact=False, rtol=1e-10, atol=1e-12)
    result = dict(passed=True, six_model_inference_parity=True,
                  same_mask_pairwise_comparisons=True, coverage_nesting=True,
                  raw_score_has_no_return_MSE=True, original_sources_unchanged=True,
                  exploratory_only=True, formal_promotion=False)
    (out / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
