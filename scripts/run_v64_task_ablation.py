"""Nine fixed task ablations; F gradients, E checkpoint selection, C/V final only."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation
from src.cuda_finite_check import noncompacting_finite_check
from src.windows_checkpoint_io import atomic_checkpoint
from src.v61_state_momentum import StateRanker
from src.v63_residual_learning import bounded_increment, fit_gain
from src.v64_task_ablation import active_tasks, activate_heads, mode_loss
from src.research_rank_metrics import rank_metrics_named
from scripts.build_v63_residual_targets import sha
from scripts.run_v61_state_ranker import infer
from scripts.run_v59_raw_mean import score
from scripts.run_v45_research_pipeline import process_alive

REPORT = ROOT / 'output/reports/task_ablation_v64'
MODELS = ROOT / 'output/models/task_ablation_v64'


def save(path, data):
    engine.write_json(path, data)


def bind(smoke):
    target = json.loads((REPORT / 'target_status.json').read_text())
    sources = json.loads((REPORT / 'target_manifest.json').read_text())
    assert sha(REPORT / 'target_manifest.json') == target['target_manifest_sha256']
    for name, expected in {**sources, **target['artifacts']}.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen target source changed: '+name)
    extra = [Path(__file__), ROOT / 'src/hierarchical_asset_itransformer.py', ROOT / 'src/v49_masked_forecast.py',
        ROOT / 'src/v61_state_momentum.py', ROOT / 'src/research_rank_metrics.py',
        ROOT / 'src/cuda_finite_check.py', ROOT / 'src/windows_checkpoint_io.py',
        ROOT / 'scripts/run_v61_state_ranker.py', ROOT / 'scripts/run_v59_raw_mean.py',
        ROOT / 'data_processed/state_momentum_v61/signals.npz', REPORT / 'target_manifest.json']
    identity = {**sources, **target['artifacts'], **{str(p.relative_to(ROOT)): sha(p) for p in extra}}
    if not smoke:
        path = REPORT / 'training_manifest.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Incompatible V64 source identity')
        save(path, identity)
    return identity, target


def load_context():
    with np.load(ROOT / 'data_processed/corporate_action_v58/dataset.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / 'data_processed/state_momentum_v61/signals.npz') as z:
        signals = {k: z[k].copy() for k in z.files}
    assert len(a['module_index']) == 38 and np.array_equal(a['dates'], signals['dates'])
    extra = np.stack([2*signals['cohort']-1, np.tanh(signals['extension']/3), signals['value_boundary']], -1)
    a['local_values'] = np.concatenate([a['local_values'], np.nan_to_num(extra)], -1).astype(np.float32)
    a['local_masks'] = np.concatenate([a['local_masks'], np.isfinite(extra)], -1)
    return a, signals


def parts_for(r):
    parts, cursor = {}, 0
    for part in ['fit', 'early', 'calibration', 'validation']:
        count = len(r[part+'_indices'])
        parts[part] = slice(cursor, cursor+count)
        cursor += count
    return parts


def raw_outputs(model, a, indices, old, device, parent, scales):
    output = infer(model, a, indices, old, device, parent['batch_size'])
    bound = parent['increment_bound_sigma']
    return {t: bound*np.tanh(v/bound)*scales[t] for t, v in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    lock = REPORT / 'active.json'
    if lock.exists() and process_alive(json.loads(lock.read_text())['pid']):
        raise RuntimeError('V64 already active')
    identity, status = bind(args.smoke)
    cfg = json.loads((ROOT / 'config/v64_task_ablation.json').read_text())
    parent = json.loads((ROOT / cfg['parent_config']).read_text())
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    a, signals = load_context()
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    device = torch.device('cuda')
    gt = {t: torch.tensor(v, device=device) for t, v in groups.items()}
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    progress = dict(pid=os.getpid(), status='training_F_selecting_E', completed=0, total=status['tasks'],
        started_at=datetime.now(timezone.utc).isoformat(), V_opened=False, formal_promotion=False, ppo_training_started=False)
    if not args.smoke:
        save(lock, dict(pid=os.getpid()))
        save(REPORT / 'progress.json', progress)
    smoke_rows, selected_rows = [], []
    years = [row['year'] for row in status['splits'] if row['eligible']]
    try:
        with training_implementation(), noncompacting_finite_check():
            for year in years:
                with np.load(ROOT / f'data_processed/task_ablation_v64/{year}.npz') as z:
                    r = {k: z[k].copy() for k in z.files}
                fit, early = r['fit_indices'], r['early_indices']
                parts = parts_for(r)
                clipped, _ = engine.clip_fold_features(a, fit, old)
                scales = {t: r[t+'_scale'] for t in groups}
                masks = {t: r[t+'_mask'].astype(bool) for t in groups}
                targets = {t: np.nan_to_num(r[t+'_residual']/scales[t]).astype(np.float32) for t in groups}
                effective = np.flatnonzero(np.logical_or.reduce([masks[t][parts['fit']].any((1, 2)) for t in groups]))
                for mode in cfg['modes']:
                    active = active_tasks(mode)
                    for seed in (cfg['seeds'][:1] if args.smoke else cfg['seeds']):
                        task_identity = dict(manifest=identity, mode=mode, year=year, seed=seed)
                        folder = MODELS / str(year) / mode / str(seed)
                        done = folder / 'selection.json'
                        if not args.smoke and done.exists():
                            selected = json.loads(done.read_text())
                            assert selected['identity'] == task_identity
                            selected_rows.append(selected)
                            progress['completed'] = len(selected_rows)
                            save(REPORT / 'progress.json', progress)
                            continue
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        model = StateRanker(engine.ForecastNetwork(clipped, old), old['d_model']).to(device)
                        active_head_parameters = activate_heads(model, mode)
                        initial_inactive = {t: copy.deepcopy(model.heads[t].state_dict()) for t in groups if t not in active}
                        optimizer = torch.optim.AdamW([dict(params=active_head_parameters, lr=parent['head_lr']),
                            dict(params=model.network.core.parameters(), lr=parent['encoder_lr'])], weight_decay=parent['weight_decay'])
                        rng = np.random.default_rng(seed)
                        first, best, best_epoch = 1, float('inf'), 0
                        best_state, logs = copy.deepcopy(model.state_dict()), []
                        current = folder / 'current.pt'
                        if not args.smoke:
                            folder.mkdir(parents=True, exist_ok=True)
                            if current.exists():
                                checkpoint = torch.load(current, map_location='cpu', weights_only=False)
                                assert checkpoint['identity'] == task_identity
                                model.load_state_dict(checkpoint['state'])
                                optimizer.load_state_dict(checkpoint['optimizer'])
                                first, best, best_epoch = checkpoint['epoch']+1, checkpoint['best'], checkpoint['best_epoch']
                                best_state, logs = checkpoint['best_state'], checkpoint['logs']
                                rng.bit_generator.state = checkpoint['numpy_rng']
                                torch.set_rng_state(checkpoint['torch_rng'].cpu())
                                torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])

                        def evaluate_early():
                            raw = raw_outputs(model, clipped, early, old, device, parent, scales)
                            baseline = {t: r[t+'_base'][parts['early']] for t in active}
                            pred = {t: baseline[t]+raw[t] for t in active}
                            truth = {t: a[t+'_raw_relative_target'][early] for t in active}
                            mask = {t: masks[t][parts['early']] for t in active}
                            return score(pred, baseline, truth, mask, groups)[0]

                        epoch = first-1
                        for epoch in range(first, (1 if args.smoke else cfg['maximum_epochs'])+1):
                            model.train()
                            order = rng.permutation(effective)
                            losses, updates = [], 0
                            for start in range(0, len(order), parent['batch_size']):
                                pos = order[start:start+parent['batch_size']]
                                m = {t: torch.tensor(masks[t][pos], device=device) for t in groups}
                                if not any(m[t].any().item() for t in active):
                                    continue
                                y = {t: torch.tensor(targets[t][pos], device=device) for t in groups}
                                optimizer.zero_grad(set_to_none=True)
                                pred = bounded_increment(model(engine.batch_at(clipped, fit[pos], old, device)), parent['increment_bound_sigma'])
                                loss = mode_loss(pred, y, m, gt, mode, parent['rank_weight'], parent['increment_penalty'])
                                if not torch.isfinite(loss):
                                    raise FloatingPointError('Nonfinite task loss')
                                loss.backward()
                                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                                if args.smoke:
                                    checks = {prefix: any(prefix in name and p.grad is not None and p.grad.abs().sum().item() > 0
                                        for name, p in model.named_parameters()) for prefix in ['local_projection', 'encoder.layers.0', 'encoder.layers.1']}
                                    checks['active_heads'] = all(any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.heads[t].parameters()) for t in active)
                                    checks['inactive_head_no_grad'] = all(p.grad is None for t in groups if t not in active for p in model.heads[t].parameters())
                                    assert all(checks.values()), checks
                                    optimizer.step()
                                    for t, before in initial_inactive.items():
                                        for k, v in model.heads[t].state_dict().items():
                                            assert torch.equal(before[k], v)
                                    early_ratio = evaluate_early()
                                    assert np.isfinite(early_ratio)
                                    smoke_rows.append(dict(mode=mode, checks=checks, passed=True))
                                    print(json.dumps(dict(mode=mode, smoke=True, checks=checks)), flush=True)
                                    break
                                optimizer.step()
                                updates += 1
                                losses.append(float(loss.detach()))
                            if args.smoke:
                                break
                            if not updates:
                                raise ValueError('No active updates')
                            if epoch == 1 or epoch % cfg['validation_every'] == 0:
                                ratio = evaluate_early()
                                if not np.isfinite(ratio):
                                    raise FloatingPointError('Nonfinite inner E ratio')
                                if ratio < best-1e-4:
                                    best, best_epoch, best_state = ratio, epoch, copy.deepcopy(model.state_dict())
                                row = dict(year=year, mode=mode, seed=seed, epoch=epoch, train_loss=float(np.mean(losses)),
                                    early_MSE_ratio=ratio, best_epoch=best_epoch, updates=updates, gradient_norm=float(grad), V_opened=False)
                                logs.append(row)
                                pd.DataFrame(logs).to_csv(folder / 'loss.csv', index=False)
                                progress['current'] = row
                                save(REPORT / 'progress.json', progress)
                                print(json.dumps(row), flush=True)
                            atomic_checkpoint(dict(identity=task_identity, epoch=epoch, state=model.state_dict(), optimizer=optimizer.state_dict(),
                                best=best, best_epoch=best_epoch, best_state=best_state, logs=logs,
                                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), current)
                            if epoch >= cfg['minimum_epochs'] and epoch-best_epoch >= cfg['patience_epochs']:
                                break
                        if args.smoke:
                            del model, optimizer
                            torch.cuda.empty_cache()
                            continue
                        for t, before in initial_inactive.items():
                            for k, v in best_state.items():
                                if k.startswith('heads.'+t+'.'):
                                    assert torch.equal(before[k.split('.', 2)[2]].cpu(), v.cpu())
                        selected = dict(identity=task_identity, year=year, mode=mode, seed=seed,
                            selected_epoch=best_epoch, trained_epochs=epoch, early_MSE_ratio=best,
                            checkpoint_chosen_on='inner_E_only', C_or_V_used_for_selection=False)
                        atomic_checkpoint(dict(identity=task_identity, state=best_state, scale=scales), folder / 'selected.pt')
                        save(done, selected)
                        selected_rows.append(selected)
                        progress['completed'] = len(selected_rows)
                        save(REPORT / 'progress.json', progress)
                        del model, optimizer
                        torch.cuda.empty_cache()
            if args.smoke:
                assert len(smoke_rows) == 3
                save(REPORT / 'smoke.json', dict(passed=True, modes=smoke_rows, C_or_V_used=False, identity=identity))
                return
            # Open C/V only after all nine E-selected checkpoints are fixed.
            assert len(selected_rows) == status['tasks']
            selected_hashes = {}
            for row in selected_rows:
                path = MODELS / str(row['year']) / row['mode'] / str(row['seed']) / 'selected.pt'
                selected_hashes[str(path.relative_to(ROOT))] = sha(path)
            save(REPORT / 'all_E_selected_before_CV.json', selected_hashes)
            progress.update(status='evaluating_fixed_models_on_CV', V_opened=True)
            save(REPORT / 'progress.json', progress)
            summary_rows, gain_rows = [], []
            for row in selected_rows:
                year, mode, seed = row['year'], row['mode'], row['seed']
                with np.load(ROOT / f'data_processed/task_ablation_v64/{year}.npz') as z:
                    r = {k: z[k].copy() for k in z.files}
                parts = parts_for(r)
                fit, ci, vi = r['fit_indices'], r['calibration_indices'], r['validation_indices']
                clipped, _ = engine.clip_fold_features(a, fit, old)
                folder = MODELS / str(year) / mode / str(seed)
                selected = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=False)
                model = StateRanker(engine.ForecastNetwork(clipped, old), old['d_model']).to(device)
                model.load_state_dict(selected['state'])
                raw = raw_outputs(model, clipped, np.concatenate([ci, vi]), old, device, parent, selected['scale'])
                means, increments, gains = {}, {}, {}
                for task in groups:
                    active = task in active_tasks(mode)
                    gain = fit_gain(raw[task][:len(ci)], r[task+'_residual'][parts['calibration']],
                        r[task+'_mask'][parts['calibration']], groups[task], ridge=parent['calibration_ridge']) if active else np.zeros_like(selected['scale'][task])
                    gain_rows.extend(dict(year=year, mode=mode, seed=seed, task=task, horizon=h,
                        group=int(g), gain=float(gain[h, groups[task] == g][0]), trained=active)
                        for h in range(3) for g in np.unique(groups[task]))
                    inc = raw[task][len(ci):]*gain if active else np.zeros_like(raw[task][len(ci):])
                    base = r[task+'_base'][parts['validation']]
                    means[task], increments[task], gains[task] = base+inc, inc, gain
                    mask = r[task+'_mask'][parts['validation']]
                    truth = a[task+'_raw_relative_target'][vi]
                    ratio, delta = score({task: means[task]}, {task: base}, {task: truth}, {task: mask}, groups)
                    _, _, ic = rank_metrics_named({task: means[task]}, {'control': {task: base}},
                        {task: truth}, {task: mask}, groups, reference='control')
                    summary_rows.append(dict(year=year, mode=mode, seed=seed, task=task, trained=active,
                        V_MSE_ratio=ratio, V_delta_rank_ic=delta, V_rank_ic=ic,
                        valid_cells=int(mask.sum()), active_entities=int(mask.any((0, 1)).sum()),
                        selected_epoch=row['selected_epoch'], trained_epochs=row['trained_epochs'], formal_promotion=False))
                np.savez_compressed(folder / 'validation_predictions.npz', **means,
                    **{t+'_increment': v for t, v in increments.items()}, **{t+'_gain': v for t, v in gains.items()})
                del model
                torch.cuda.empty_cache()
            bind(False)
            for name, expected in selected_hashes.items():
                assert sha(ROOT / name) == expected
            pd.DataFrame(summary_rows).to_csv(REPORT / 'V_task_results.csv', index=False)
            pd.DataFrame(gain_rows).to_csv(REPORT / 'calibration_gains.csv', index=False)
            pd.DataFrame([{k: v for k, v in row.items() if k != 'identity'} for row in selected_rows]).to_csv(REPORT / 'selection_summary.csv', index=False)
            progress.update(status='completed_diagnostic_only', formal_promotion=False, ppo_training_started=False)
            save(REPORT / 'progress.json', progress)
            print(json.dumps(dict(completed=True, tasks=len(selected_rows), formal_promotion=False)), flush=True)
    except BaseException as exc:
        if not args.smoke:
            progress.update(status='failed', error=type(exc).__name__+': '+str(exc))
            save(REPORT / 'progress.json', progress)
        raise
    finally:
        if not args.smoke and lock.exists() and json.loads(lock.read_text())['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
