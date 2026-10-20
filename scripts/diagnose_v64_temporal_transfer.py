"""Post-run E/C/V transfer diagnostic; does not train, select, flip or promote."""
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
from src.v61_state_momentum import StateRanker
from src.v64_task_ablation import active_tasks
from src.temporal_transfer_diagnostics import residual_alignment
from scripts.run_v64_task_ablation import bind, load_context, parts_for, raw_outputs, MODELS, REPORT
from scripts.build_v63_residual_targets import sha
from scripts.diagnose_v61_completed import daily_rank


def main():
    identity, target = bind(True)  # Read-only provenance check; do not rewrite training manifest.
    assert identity == json.loads((REPORT / 'training_manifest.json').read_text())
    assert json.loads((REPORT / 'artifact_audit.json').read_text())['passed']
    cfg = json.loads((ROOT / 'config/v64_task_ablation.json').read_text())
    parent = json.loads((ROOT / cfg['parent_config']).read_text())
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    a, signals = load_context()
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    names = ['rates', 'credit', 'convertible', 'equity', 'alternatives']
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('Use CUDA inference runtime')
    rows, matched, artifacts = [], [], {}
    with training_implementation(), noncompacting_finite_check(), torch.no_grad():
        for definition in target['splits']:
            if not definition['eligible']:
                continue
            year = definition['year']
            with np.load(ROOT / f'data_processed/task_ablation_v64/{year}.npz') as z:
                r = {k: z[k].copy() for k in z.files}
            parts = parts_for(r)
            fit = r['fit_indices']
            indices = np.concatenate([r[p+'_indices'] for p in ['early', 'calibration', 'validation']])
            clipped, _ = engine.clip_fold_features(a, fit, old)
            for mode in cfg['modes']:
                for seed in cfg['seeds']:
                    folder = MODELS / str(year) / mode / str(seed)
                    source = folder / 'selected.pt'
                    artifacts[str(source.relative_to(ROOT))] = sha(source)
                    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
                    assert checkpoint['identity']['manifest'] == identity
                    model = StateRanker(engine.ForecastNetwork(clipped, old), old['d_model']).to(device)
                    model.load_state_dict(checkpoint['state'])
                    raw = raw_outputs(model, clipped, indices, old, device, parent, checkpoint['scale'])
                    with np.load(folder / 'validation_predictions.npz') as z:
                        stored = {k: z[k].copy() for k in z.files}
                    artifacts[str((folder / 'validation_predictions.npz').relative_to(ROOT))] = sha(folder / 'validation_predictions.npz')
                    offset = 0
                    for part in ['early', 'calibration', 'validation']:
                        ix = r[part+'_indices']
                        rp = slice(offset, offset+len(ix))
                        for task in active_tasks(mode):
                            m, y = r[task+'_mask'][parts[part]], r[task+'_residual'][parts[part]]
                            base = r[task+'_base'][parts[part]]
                            truth = a[task+'_raw_relative_target'][ix]
                            p = raw[task][rp]
                            if part == 'validation':
                                np.testing.assert_allclose(p*stored[task+'_gain'], stored[task+'_increment'], rtol=1e-4, atol=1e-6)
                            for h, horizon in enumerate([5, 20, 60]):
                                for group in np.unique(groups[task]):
                                    members = groups[task] == group
                                    mask = m[:, h, members]
                                    x, residual = p[:, h, members], y[:, h, members]
                                    stats = residual_alignment(x, residual, mask)
                                    raw_ic = engine.finite_mean(daily_rank(x, residual, mask))
                                    gain = float(stored[task+'_gain'][h, members][0])
                                    rows.append(dict(year=year, mode=mode, seed=seed, task=task,
                                        group=names[int(group)] if task == 'asset' else 'cross_sleeve', horizon=horizon,
                                        part=part, C_gain=gain, residual_rank_ic=raw_ic,
                                        **stats, prediction='uncalibrated_raw_increment',
                                        C_gain_applied=False))
                            if part == 'validation':
                                # Relative baseline comparisons use common support per baseline only.
                                controls = dict(historical_control=base)
                                for name in ['momentum20', 'cohort', 'price_soft']:
                                    x, weights = signals[name], a['sleeve_reference_weights']
                                    if task == 'asset':
                                        value = x
                                    else:
                                        value = np.sum(weights*np.where(np.isfinite(x[:, None]), x[:, None], 0), -1)
                                        available = (weights.sum(-1)>0) & ((weights == 0) | np.isfinite(x[:, None])).all(-1)
                                        value = np.where(available, value, np.nan)
                                    controls[name] = np.repeat(value[:, None], 3, 1)[ix]
                                for name, control in controls.items():
                                    for h, horizon in enumerate([5, 20, 60]):
                                        for group in np.unique(groups[task]):
                                            members = groups[task] == group
                                            pred = stored[task][:, h, members]
                                            yy, b = truth[:, h, members], control[:, h, members]
                                            good = m[:, h, members] & np.isfinite(pred) & np.isfinite(b) & np.isfinite(yy)
                                            ic, bic = daily_rank(pred, yy, good), daily_rank(b, yy, good)
                                            matched.append(dict(year=year, mode=mode, seed=seed, task=task,
                                                group=names[int(group)] if task == 'asset' else 'cross_sleeve', horizon=horizon,
                                                baseline=name, prediction_rank_ic=engine.finite_mean(ic),
                                                baseline_rank_ic=engine.finite_mean(bic), delta_rank_ic=engine.finite_mean(ic-bic),
                                                matched_rank_dates=int(np.isfinite(ic-bic).sum()), valid_cells=int(good.sum())))
                        offset += len(ix)
                    del model, checkpoint
                    torch.cuda.empty_cache()
                    print(json.dumps(dict(mode=mode, seed=seed, diagnosed=True)), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT / 'temporal_transfer_components.csv', index=False)
    pd.DataFrame(matched).to_csv(REPORT / 'matched_baseline_components.csv', index=False)
    # Sign changes are descriptive: E was selected and C was used for calibration.
    wide = frame.pivot(index=['year', 'mode', 'seed', 'task', 'group', 'horizon'], columns='part', values='alignment').reset_index()
    wide['C_positive_V_negative'] = (wide.calibration > 0) & (wide.validation < 0)
    wide['E_positive_C_nonpositive'] = (wide.early > 0) & (wide.calibration <= 0)
    wide.to_csv(REPORT / 'alignment_transitions.csv', index=False)
    pd.DataFrame(matched).groupby(['year', 'mode', 'task', 'baseline'], as_index=False)[
        ['prediction_rank_ic', 'baseline_rank_ic', 'delta_rank_ic']].mean().to_csv(REPORT / 'matched_baseline_summary.csv', index=False)
    final_identity, _ = bind(True)
    assert identity == final_identity
    for name, expected in artifacts.items():
        assert sha(ROOT / name) == expected
    result = dict(completed=True, models_inferred=9, V_increment_reproduced=True,
        trained=False, checkpoint_selection_changed=False, V_direction_flips_applied=False,
        calibration_applied_to_E_or_C=False, formal_promotion=False, explanatory_only=True,
        component_count=len(wide), C_positive_V_negative=int(wide.C_positive_V_negative.sum()),
        E_positive_C_nonpositive=int(wide.E_positive_C_nonpositive.sum()),
        script_sha256=sha(Path(__file__)), helper_sha256=sha(ROOT / 'src/temporal_transfer_diagnostics.py'))
    (REPORT / 'temporal_transfer_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (REPORT / 'temporal_transfer_sources.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
