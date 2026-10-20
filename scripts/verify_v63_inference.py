"""Re-infer selected C/V checkpoints and independently reconstruct calibrated views."""
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
from src.v63_residual_learning import fit_gain
from scripts.run_v61_state_ranker import infer
from scripts.build_v63_residual_targets import sha


def verify_manifest(manifest):
    for name, expected in manifest.items():
        path = ROOT / ('data_processed/corporate_action_v58/dataset.npz' if name == 'dataset_sha256' else name)
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + name)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Use GPU runtime to reproduce selected-checkpoint inference')
    report = ROOT / 'output/reports/residual_learning_v63'
    manifest = json.loads((report / 'training_manifest.json').read_text())
    verify_manifest(manifest)
    cfg = json.loads((ROOT / 'config/v63_residual_learning.json').read_text())
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    with np.load(ROOT / 'data_processed/corporate_action_v58/dataset.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / 'data_processed/state_momentum_v61/signals.npz') as z:
        s = {k: z[k].copy() for k in z.files}
    features = np.stack([2*s['cohort']-1, np.tanh(s['extension']/3), s['value_boundary']], -1)
    a['local_values'] = np.concatenate([a['local_values'], np.nan_to_num(features)], -1).astype(np.float32)
    a['local_masks'] = np.concatenate([a['local_masks'], np.isfinite(features)], -1)
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('cuda')
    rows, curves, artifacts = [], [], {}
    with training_implementation(), noncompacting_finite_check(), torch.no_grad():
        for year in cfg['years']:
            with np.load(ROOT / f'data_processed/residual_learning_v63/{year}.npz') as z:
                r = {k: z[k].copy() for k in z.files}
            ti, ci, vi = [r[k+'_indices'] for k in ['train', 'calibration', 'validation']]
            cp, vp = slice(len(ti), len(ti)+len(ci)), slice(len(ti)+len(ci), None)
            clipped, _ = engine.clip_fold_features(a, ti, old)
            for seed in cfg['seeds']:
                folder = ROOT / f'output/models/residual_learning_v63/{year}/{seed}'
                selected = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
                assert selected['identity']['manifest'] == manifest
                model = StateRanker(engine.ForecastNetwork(clipped, old), old['d_model']).to(device)
                model.load_state_dict(selected['state'])
                outputs = infer(model, clipped, np.concatenate([ci, vi]), old, device, cfg['batch_size'])
                with np.load(folder / 'validation_predictions.npz') as z:
                    stored = {k: z[k].copy() for k in z.files}
                for task in groups:
                    sigma = r[task+'_scale']
                    raw = cfg['increment_bound_sigma']*np.tanh(outputs[task]/cfg['increment_bound_sigma'])*sigma
                    gain = fit_gain(raw[:len(ci)], r[task+'_residual'][cp], r[task+'_mask'][cp], groups[task], ridge=cfg['calibration_ridge'])
                    np.testing.assert_allclose(gain, selected['gain'][task], rtol=1e-4, atol=1e-6)
                    increment = raw[len(ci):]*gain
                    predicted = r[task+'_base'][vp]+increment
                    np.testing.assert_allclose(increment, stored[task+'_increment'], rtol=1e-4, atol=1e-6)
                    np.testing.assert_allclose(predicted, stored[task], rtol=1e-4, atol=1e-6, equal_nan=True)
                    for h, horizon in enumerate([5, 20, 60]):
                        for group in np.unique(groups[task]):
                            members = groups[task] == group
                            m = r[task+'_mask'][vp][:, h, members]
                            p, y = increment[:, h, members], r[task+'_residual'][vp][:, h, members]
                            good = m & np.isfinite(p) & np.isfinite(y)
                            before = np.mean(y[good]**2) if good.any() else np.nan
                            after = np.mean((p[good]-y[good])**2) if good.any() else np.nan
                            rows.append(dict(year=year, seed=seed, task=task, group_index=int(group), horizon=horizon,
                                gain=float(gain[h, members][0]),
                                matched_label_cells=int(good.sum()),
                                pooled_cell_MSE_ratio=after/before if before > 1e-16 else np.nan,
                                inference_gain_max_abs_error=float(np.max(np.abs(gain[h, members]-selected['gain'][task][h, members]))),
                                inference_increment_max_abs_error=float(np.max(np.abs(increment[:, h, members]-stored[task+'_increment'][:, h, members])))))
                summary = json.loads((folder / 'summary.json').read_text())
                loss = pd.read_csv(folder / 'loss.csv')
                best = loss[loss.epoch == summary['selected_epoch']].iloc[0]
                first, last = loss.iloc[0], loss.iloc[-1]
                curves.append(dict(year=year, seed=seed, selected_epoch=summary['selected_epoch'], trained_epochs=summary['trained_epochs'],
                    first_train_loss=float(first.train_loss), selected_train_loss=float(best.train_loss), last_train_loss=float(last.train_loss),
                    selected_validation_MSE_ratio=summary['validation_MSE_ratio'], last_validation_MSE_ratio=float(last.validation_MSE_ratio),
                    validation_deterioration_after_selection=float(last.validation_MSE_ratio-summary['validation_MSE_ratio'])))
                for name in ['selected.pt', 'validation_predictions.npz', 'summary.json', 'loss.csv']:
                    artifacts[str((folder/name).relative_to(ROOT))] = sha(folder/name)
                print(json.dumps(dict(year=year, seed=seed, C_only_gain_reproduced=True, V_inference_reproduced=True)), flush=True)
                del model, selected
                torch.cuda.empty_cache()
    verify_manifest(manifest)
    pd.DataFrame(rows).to_csv(report / 'calibration_inference_diagnostics.csv', index=False)
    pd.DataFrame(curves).to_csv(report / 'learning_curve_diagnostics.csv', index=False)
    (report / 'inference_source_hashes.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    frame = pd.DataFrame(rows)
    result = dict(passed=True, selected_models_reinferred=6, calibration_fitted_only_on_C=True,
        stored_V_means_and_increments_reproduced=True, positive_component_gains=int((frame.gain > 0).sum()),
        total_component_gains=len(frame), retrained=False, selection_changed=False, G_or_test_used=False,
        verification_script_sha256=sha(Path(__file__)), formal_promotion=False)
    (report / 'inference_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
