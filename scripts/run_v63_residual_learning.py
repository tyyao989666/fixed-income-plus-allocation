"""Six fixed full-encoder residual experiments; historical V research only."""
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
from src.v63_residual_learning import bounded_increment, residual_loss, fit_gain
from scripts.build_v63_residual_targets import sha
from scripts.run_v61_state_ranker import infer, rank_metrics
from scripts.run_v59_raw_mean import score
from scripts.run_v45_research_pipeline import process_alive

REPORT = ROOT / 'output/reports/residual_learning_v63'
MODELS = ROOT / 'output/models/residual_learning_v63'


def save(path, obj):
    engine.write_json(path, obj)


def verify_and_bind(smoke):
    target = json.loads((REPORT / 'target_status.json').read_text())
    files = [Path(__file__), ROOT / 'src/v63_residual_learning.py', ROOT / 'src/v61_state_momentum.py',
        ROOT / 'src/v45_forecast.py', ROOT / 'src/v49_masked_forecast.py',
        ROOT / 'src/hierarchical_asset_itransformer.py', ROOT / 'src/cuda_finite_check.py',
        ROOT / 'src/windows_checkpoint_io.py', ROOT / 'scripts/run_v61_state_ranker.py',
        ROOT / 'scripts/run_v59_raw_mean.py', ROOT / 'scripts/build_v63_residual_targets.py',
        ROOT / 'config/v63_residual_learning.json', ROOT / 'config/v45_research.yaml',
        ROOT / 'docs/v63_residual_learning_protocol.md', REPORT / 'target_source_manifest.json']
    original = json.loads((REPORT / 'target_source_manifest.json').read_text())
    for name, expected in {**original, **target['targets']}.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Target provenance changed: ' + name)
    if sha(REPORT / 'target_source_manifest.json') != target['source_manifest_sha256']:
        raise ValueError('Target manifest changed')
    files += [ROOT / name for name in target['targets']]
    files += [ROOT / 'data_processed/state_momentum_v61/signals.npz']
    identity = {str(path.relative_to(ROOT)): sha(path) for path in files}
    identity['dataset_sha256'] = target['dataset_sha256']
    assert sha(ROOT / 'data_processed/corporate_action_v58/dataset.npz') == identity['dataset_sha256']
    if not smoke:
        manifest = REPORT / 'training_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Training identity changed; incompatible resume refused')
        save(manifest, identity)
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA interpreter required')
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = REPORT / 'active.json'
    if lock.exists() and process_alive(json.loads(lock.read_text())['pid']):
        raise RuntimeError('V63 already running')
    manifest = verify_and_bind(args.smoke)
    cfg = json.loads((ROOT / 'config/v63_residual_learning.json').read_text())
    old = engine.configuration(ROOT / 'config/v45_research.yaml')
    with np.load(ROOT / 'data_processed/corporate_action_v58/dataset.npz') as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / 'data_processed/state_momentum_v61/signals.npz') as z:
        s = {k: z[k].copy() for k in z.files}
    assert len(a['module_index']) == 38 and np.array_equal(a['dates'], s['dates'])
    features = np.stack([2*s['cohort']-1, np.tanh(s['extension']/3), s['value_boundary']], -1)
    a['local_values'] = np.concatenate([a['local_values'], np.nan_to_num(features)], -1).astype(np.float32)
    a['local_masks'] = np.concatenate([a['local_masks'], np.isfinite(features)], -1)
    groups = dict(asset=a['module_index'], sleeve=np.zeros(5, int))
    device = torch.device('cuda')
    gt = {t: torch.tensor(g, device=device) for t, g in groups.items()}
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    progress = dict(status='training', pid=os.getpid(), completed=0, total=6,
        started_at=datetime.now(timezone.utc).isoformat(), formal_promotion=False, ppo_training_started=False)
    if not args.smoke:
        save(lock, dict(pid=os.getpid()))
        save(REPORT / 'progress.json', progress)
    summaries, references, ensembles = [], {}, {}
    try:
        with training_implementation(), noncompacting_finite_check():
            for year in cfg['years']:
                with np.load(ROOT / f'data_processed/residual_learning_v63/{year}.npz') as z:
                    r = {k: z[k].copy() for k in z.files}
                ti, ci, vi = [r[k + '_indices'] for k in ['train', 'calibration', 'validation']]
                assert np.array_equal(r['dates'], a['dates'][np.concatenate([ti, ci, vi])])
                clipped, _ = engine.clip_fold_features(a, ti, old)
                parts = dict(train=slice(0, len(ti)), calibration=slice(len(ti), len(ti)+len(ci)), validation=slice(len(ti)+len(ci), None))
                scale = {t: r[t + '_scale'] for t in groups}
                target = {t: np.nan_to_num(r[t + '_residual']/scale[t], nan=0., posinf=0., neginf=0.).astype(np.float32) for t in groups}
                masks = {t: r[t + '_mask'].astype(bool) for t in groups}
                vp = parts['validation']
                truth = {t: a[t + '_raw_relative_target'][vi] for t in groups}
                baseline = {t: r[t + '_base'][vp] for t in groups}
                vm = {t: masks[t][vp] for t in groups}
                controls = {'historical_control': baseline}
                for name in ['momentum20', 'cohort', 'price_soft']:
                    x, w = s[name], a['sleeve_reference_weights']
                    sleeve = np.sum(w*np.where(np.isfinite(x[:, None]), x[:, None], 0), -1)
                    good = (w.sum(-1) > 0) & ((w == 0) | np.isfinite(x[:, None])).all(-1)
                    controls[name] = dict(asset=np.repeat(x[:, None], 3, 1)[vi], sleeve=np.repeat(np.where(good, sleeve, np.nan)[:, None], 3, 1)[vi])
                common = {t: vm[t].copy() for t in groups}
                for base in controls.values():
                    for t in groups:
                        common[t] &= np.isfinite(base[t])
                references[year] = dict(truth=truth, masks=vm, common=common, baseline=baseline, controls=controls)
                for seed in cfg['seeds']:
                    folder = MODELS / str(year) / str(seed)
                    identity = dict(manifest=manifest, year=year, seed=seed)
                    done = folder / 'summary.json'
                    if not args.smoke and done.exists():
                        summary = json.loads(done.read_text())
                        if summary['identity'] != identity:
                            raise ValueError('Completed model identity mismatch')
                    else:
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        model = StateRanker(engine.ForecastNetwork(clipped, old), old['d_model']).to(device)
                        optimizer = torch.optim.AdamW([dict(params=model.heads.parameters(), lr=cfg['head_lr']),
                            dict(params=model.network.core.parameters(), lr=cfg['encoder_lr'])], weight_decay=cfg['weight_decay'])
                        rng = np.random.default_rng(seed)
                        first, best, best_epoch = 1, float('inf'), 0
                        best_state, logs = copy.deepcopy(model.state_dict()), []
                        current = folder / 'current.pt'
                        if not args.smoke:
                            folder.mkdir(parents=True, exist_ok=True)
                            if current.exists():
                                checkpoint = torch.load(current, map_location='cpu', weights_only=False)
                                assert checkpoint['identity'] == identity
                                model.load_state_dict(checkpoint['state'])
                                optimizer.load_state_dict(checkpoint['optimizer'])
                                first, best, best_epoch = checkpoint['epoch']+1, checkpoint['best'], checkpoint['best_epoch']
                                best_state, logs = checkpoint['best_state'], checkpoint['logs']
                                rng.bit_generator.state = checkpoint['numpy_rng']
                                torch.set_rng_state(checkpoint['torch_rng'].cpu())
                                torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])

                        def evaluate():
                            scores = infer(model, clipped, np.concatenate([ci, vi]), old, device, cfg['batch_size'])
                            gains, means, increments = {}, {}, {}
                            for t in groups:
                                raw = cfg['increment_bound_sigma']*np.tanh(scores[t]/cfg['increment_bound_sigma'])*scale[t]
                                gains[t] = fit_gain(raw[:len(ci)], r[t + '_residual'][parts['calibration']],
                                    masks[t][parts['calibration']], groups[t], ridge=cfg['calibration_ridge'])
                                increments[t] = raw[len(ci):]*gains[t]
                                means[t] = baseline[t]+increments[t]
                            ratio, delta = score(means, baseline, truth, vm, groups)
                            # Calibration residual ranking is separate from raw-return ranking.
                            residual_ic = []
                            for t in groups:
                                for h in range(3):
                                    for g in np.unique(groups[t]):
                                        members = groups[t] == g
                                        residual_ic.append(engine.finite_mean(engine.daily_ic(increments[t][:, h, members],
                                            r[t + '_residual'][vp][:, h, members], vm[t][:, h, members])))
                            return ratio, delta, engine.finite_mean(residual_ic), gains, means, increments

                        effective = np.flatnonzero(np.logical_or.reduce([masks[t][parts['train']].any((1, 2)) for t in groups]))
                        if not len(effective):
                            raise ValueError('No valid training residuals')
                        for epoch in range(first, (1 if args.smoke else cfg['maximum_epochs'])+1):
                            model.train()
                            order = rng.permutation(effective)
                            losses = []
                            for start in range(0, len(order), cfg['batch_size']):
                                positions = order[start:start+cfg['batch_size']]
                                y = {t: torch.tensor(target[t][positions], device=device) for t in groups}
                                m = {t: torch.tensor(masks[t][positions], device=device) for t in groups}
                                optimizer.zero_grad(set_to_none=True)
                                pred = bounded_increment(model(engine.batch_at(clipped, ti[positions], old, device)), cfg['increment_bound_sigma'])
                                loss = residual_loss(pred, y, m, gt, cfg['rank_weight'], cfg['increment_penalty'])
                                if not torch.isfinite(loss):
                                    raise FloatingPointError('Nonfinite residual loss')
                                loss.backward()
                                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                                if args.smoke:
                                    gradients = {name: float(p.grad.abs().sum()) for name, p in model.named_parameters() if p.grad is not None}
                                    checks = {name: any(prefix in k and v > 0 for k, v in gradients.items()) for name, prefix in [
                                        ('local', 'local_projection'), ('layer_0', 'encoder.layers.0'), ('layer_1', 'encoder.layers.1'), ('heads', 'heads.')]}
                                    if not all(checks.values()):
                                        raise ValueError('Missing full encoder gradient: ' + str(checks))
                                    smoke_ratio, _, _, smoke_gains, smoke_means, _ = evaluate()
                                    if not np.isfinite(smoke_ratio) or not all(np.isfinite(smoke_means[t][vm[t]]).all() for t in groups):
                                        raise ValueError('Smoke calibration/validation path failed')
                                    save(REPORT / 'smoke.json', dict(passed=True, cuda=True, checks=checks, loss=float(loss.detach()),
                                        calibration_path_passed=True, train_batch_dates=len(positions), training_manifest=manifest, formal_promotion=False))
                                    print(json.dumps(dict(smoke_passed=True, checks=checks, loss=float(loss.detach()))), flush=True)
                                    return
                                optimizer.step()
                                losses.append(float(loss.detach()))
                            if epoch == 1 or epoch % cfg['validation_every'] == 0:
                                ratio, delta, ric, _, _, _ = evaluate()
                                if not np.isfinite(ratio):
                                    raise FloatingPointError('Nonfinite validation ratio')
                                if ratio < best-1e-4:
                                    best, best_epoch, best_state = ratio, epoch, copy.deepcopy(model.state_dict())
                                row = dict(year=year, seed=seed, epoch=epoch, train_loss=float(np.mean(losses)),
                                    validation_MSE_ratio=ratio, validation_delta_rank_ic=delta,
                                    validation_residual_ic=ric if np.isfinite(ric) else None,
                                    gradient_norm=float(grad), best_epoch=best_epoch)
                                logs.append(row)
                                pd.DataFrame(logs).to_csv(folder / 'loss.csv', index=False)
                                progress['current'] = row
                                save(REPORT / 'progress.json', progress)
                                print(json.dumps(row), flush=True)
                            atomic_checkpoint(dict(identity=identity, state=model.state_dict(), optimizer=optimizer.state_dict(),
                                epoch=epoch, best=best, best_epoch=best_epoch, best_state=best_state, logs=logs,
                                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), current)
                            if epoch >= cfg['minimum_epochs'] and epoch-best_epoch >= cfg['patience_epochs']:
                                break
                        model.load_state_dict(best_state)
                        ratio, delta, ric, gains, means, increments = evaluate()
                        summary = dict(identity=identity, year=year, seed=seed, trained_epochs=epoch, selected_epoch=best_epoch,
                            validation_MSE_ratio=ratio, validation_delta_rank_ic=delta,
                            validation_residual_ic=ric if np.isfinite(ric) else None,
                            validation_not_tradeable=True, formal_promotion=False)
                        atomic_checkpoint(dict(identity=identity, state=model.state_dict(), gain=gains, scale=scale), folder / 'selected.pt')
                        np.savez_compressed(folder / 'validation_predictions.npz', **means,
                            **{t + '_increment': increments[t] for t in groups})
                        save(done, summary)
                        del model, optimizer
                        torch.cuda.empty_cache()
                    with np.load(folder / 'validation_predictions.npz') as z:
                        ensembles.setdefault(year, []).append({k: z[k].copy() for k in z.files})
                    summaries.append(summary)
                    progress['completed'] = len(summaries)
                    save(REPORT / 'progress.json', progress)
        checks = dict(all_seed_MSE_nonworse=all(s['validation_MSE_ratio'] <= cfg['maximum_MSE_ratio']+1e-7 for s in summaries))
        details = []
        for year in cfg['years']:
            ref, preds = references[year], ensembles[year]
            mean = {t: np.mean([p[t] for p in preds], 0) for t in groups}
            # Retain all-baseline support for the fixed continuation gate, separate from own-universe MSE.
            metric, deltas, ic = rank_metrics(mean, ref['controls'], ref['truth'], ref['common'], groups)
            checks[str(year) + '_ensemble_rank_increment'] = metric >= cfg['minimum_rank_increment']
            agreement = {}
            for t in groups:
                correlations = []
                for left in range(3):
                    for right in range(left+1, 3):
                        for h in range(3):
                            for g in np.unique(groups[t]):
                                ix = groups[t] == g
                                correlations.append(engine.finite_mean(engine.daily_ic(preds[left][t+'_increment'][:, h, ix],
                                    preds[right][t+'_increment'][:, h, ix], ref['masks'][t][:, h, ix])))
                value = engine.finite_mean(correlations)
                agreement[t] = value if np.isfinite(value) else None
                checks[f'{year}_{t}_increment_agreement'] = bool(np.isfinite(value) and value >= cfg['minimum_'+t+'_seed_agreement'])
            ratio, _ = score(mean, ref['baseline'], ref['truth'], ref['masks'], groups)
            details.append(dict(year=year, own_support_MSE_ratio=ratio, common_support_rank_ic=ic,
                min_rank_increment=metric, delta_by_baseline=deltas, increment_seed_agreement=agreement))
        verify_and_bind(False)
        decision = dict(checks=checks, details=details, research_followup_allowed=all(checks.values()),
            formal_promotion=False, ppo_training_started=False, G_or_test_used=False, exploratory_validation=True,
            next_step='stop and review; no automatic search')
        save(REPORT / 'decision.json', decision)
        pd.DataFrame([{k: v for k, v in s.items() if k != 'identity'} for s in summaries]).to_csv(REPORT / 'training_summary.csv', index=False)
        progress.update(status='completed_validation_only', research_followup_allowed=decision['research_followup_allowed'])
        save(REPORT / 'progress.json', progress)
        print(json.dumps(decision), flush=True)
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
