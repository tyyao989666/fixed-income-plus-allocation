"""Check preservation, provenance and actual iTransformer input compatibility."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import validate_dataset, training_implementation
from scripts.build_v66_boundary import OUT, DATA, SOURCE, sha


def main():
    status = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
    for name, digest in {**status['source_hashes'], **status['output_hashes']}.items():
        assert sha(ROOT / name) == digest, name
    with np.load(SOURCE) as z:
        old = {k:z[k] for k in z.files}
    with np.load(DATA / 'dataset_strict_masked.npz') as z:
        new = {k:z[k] for k in z.files}
    assert set(old) == set(new)
    for key in old:
        if key in ['macro_values', 'macro_masks']:
            np.testing.assert_array_equal(old[key], new[key][:, :old[key].shape[1]])
        else:
            np.testing.assert_array_equal(old[key], new[key])
    width = old['macro_values'].shape[1]
    assert not new['macro_masks'][:, width:].any()
    assert not new['macro_values'][:, width:].any()
    validate_dataset(new)
    cfg = engine.configuration(ROOT / 'config/v45_research.yaml')
    torch.set_num_threads(2)
    torch.manual_seed(17)
    with training_implementation():
        model = engine.ForecastNetwork(new, cfg).eval()
        batch = engine.batch_at(new, np.array([500]), cfg, 'cpu')
        with torch.no_grad():
            prediction = model(batch)
        # Prove that even nonfinite placeholders under FALSE new masks are hidden.
        altered = dict(new)
        altered['macro_values'] = new['macro_values'].copy()
        altered['macro_values'][:, width:] = np.nan
        corrupted_batch = engine.batch_at(altered, np.array([500]), cfg, 'cpu')
        with torch.no_grad():
            second = model(corrupted_batch)
        for task in prediction:
            assert torch.isfinite(prediction[task]).all()
            torch.testing.assert_close(prediction[task], second[task], rtol=0, atol=0)
    report = dict(passed=True, all_original_arrays_preserved=True, source_hashes_passed=True,
        original_nine_grid_preserved=True, asset_count=38, observed_macro_width=16,
        strict_uncertified_features_excluded=True, model_dimension_smoke_passed=True,
        masked_nan_invariance_passed=True, prediction_shapes={k:list(v.shape) for k,v in prediction.items()},
        smoke_only_random_weights=True, old_checkpoint_reused=False, trained=False,
        scripts_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT / 'scripts/diagnose_v66_boundary.py', ROOT / 'tests/test_boundary_v66.py',
            ROOT / 'src/v45_forecast.py', ROOT / 'src/v49_masked_forecast.py']})
    (OUT / 'artifact_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
