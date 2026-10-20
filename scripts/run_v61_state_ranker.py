"""Fresh full-core supervised ranker, isolated calibration, fixed baseline controls."""
import copy
import json
import os
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
from src.windows_checkpoint_io import atomic_checkpoint
from src.v60_supervised_adapter import fit_prior
from src.v61_state_momentum import StateRanker, rank_loss, nested_split, fit_calibrator
from scripts.run_v59_raw_mean import sha, save, score
from scripts.run_v45_research_pipeline import process_alive


@torch.no_grad()
def infer(model, a, ix, cfg, device, batch):
    model.eval()
    result = {t:[] for t in ["asset", "sleeve"]}
    for start in range(0, len(ix), batch):
        values = model(engine.batch_at(a, ix[start:start+batch], cfg, device))
        for t,v in values.items():
            result[t].append(v.cpu().numpy())
    return {t:np.concatenate(v) for t,v in result.items()}


def rank_metrics(pred, baselines, truth, masks, groups):
    def ranked(p, y, m):
        value = engine.daily_ic(p, y, m)
        # A constant forecast has no ranking information, not missing coverage.
        for i in range(len(value)):
            good = m[i] & np.isfinite(p[i]) & np.isfinite(y[i])
            if good.sum() >= 2 and np.ptp(y[i, good]) > 0 and np.ptp(p[i, good]) == 0:
                value[i] = 0.
        return value
    deltas, ic_values = {}, []
    for name, base in baselines.items():
        terms = []
        for task in pred:
            for h in range(3):
                by_group, ranks = [], []
                for g in np.unique(groups[task]):
                    members = groups[task] == g
                    m = masks[task][:, h, members]
                    p, y, b = pred[task][:, h, members], truth[task][:, h, members], base[task][:, h, members]
                    x = ranked(p, y, m)
                    z = ranked(b, y, m)
                    by_group.append(engine.finite_mean(x-z))
                    ranks.append(engine.finite_mean(x))
                terms.append(engine.finite_mean(by_group))
                if name == "prior":
                    ic_values.extend(ranks)
        deltas[name] = engine.finite_mean(terms)
    return min(deltas.values()) if all(np.isfinite(list(deltas.values()))) else -float("inf"), deltas, engine.finite_mean(ic_values)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA interpreter required")
    cfg = json.loads((ROOT / "config/v61_state_momentum.json").read_text())
    out = ROOT / "output/reports/state_momentum_v61"
    models = ROOT / "output/models/state_momentum_v61"
    lock = out / "active.json"
    if lock.exists() and process_alive(json.loads(lock.read_text())["pid"]):
        raise RuntimeError("V61 already active")
    files = [Path(__file__), ROOT / "src/v61_state_momentum.py", ROOT / "src/v60_supervised_adapter.py",
        ROOT / "src/v59_raw_mean.py", ROOT / "scripts/run_v59_raw_mean.py", ROOT / "src/v45_forecast.py",
        ROOT / "src/v49_masked_forecast.py", ROOT / "src/hierarchical_asset_itransformer.py",
        ROOT / "src/cuda_finite_check.py", ROOT / "src/windows_checkpoint_io.py", ROOT / "config/v45_research.yaml",
        ROOT / "config/v61_state_momentum.json", ROOT / "docs/v61_state_momentum_preregistration.md",
        ROOT / "scripts/build_v61_state_signals.py", ROOT / "output/reports/state_momentum_v61/signal_status.json",
        ROOT / "data_processed/state_momentum_v61/signals.npz"]
    manifest = {str(p.relative_to(ROOT)):sha(p) for p in files}
    data = ROOT / "data_processed/corporate_action_v58/dataset.npz"
    manifest["dataset_sha256"] = sha(data)
    registry = json.loads((ROOT / "output/reports/corporate_action_v58/status.json").read_text())
    assert manifest["dataset_sha256"] == registry["dataset_sha256"]
    frozen = out / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise ValueError("Frozen V61 identity changed")
    save(frozen, manifest)
    with np.load(data) as z:
        a = {k:z[k].copy() for k in z.files}
    with np.load(ROOT / "data_processed/state_momentum_v61/signals.npz") as z:
        signals = {k:z[k].copy() for k in z.files}
    assert len(a["module_index"]) == 38 and np.array_equal(signals["dates"], a["dates"])
    extras = [2*signals["cohort"]-1, np.tanh(signals["extension"]/3), signals["value_boundary"]]
    values = np.stack(extras, -1)
    a["local_values"] = np.concatenate([a["local_values"], np.nan_to_num(values)], -1).astype(np.float32)
    a["local_masks"] = np.concatenate([a["local_masks"], np.isfinite(values)], -1)
    old = engine.configuration(ROOT / "config/v45_research.yaml")
    old["seed_workers"] = 1
    groups = dict(asset=a["module_index"], sleeve=np.zeros(5, int))
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    gt = {t:torch.tensor(v, device=device) for t,v in groups.items()}
    progress = dict(status="training", pid=os.getpid(), completed=0, total=6,
        formal_promotion=False, ppo_training_started=False, boundary_enabled=False, fresh_full_encoder=True)
    save(lock, dict(pid=os.getpid()))
    save(out / "progress.json", progress)
    rows, summaries, predictions, references = [], [], {}, {}
    try:
        with training_implementation(), noncompacting_finite_check():
            for year in cfg["years"]:
                split = nested_split(a, engine.shared_split(a, year, old), cfg["calibration_days"], cfg["minimum_inner_train_days"])
                ti, ci, vi = [split[p] for p in ["train", "calibration", "validation"]]
                clipped, _ = engine.clip_fold_features(a, ti, old)
                save(out / f"split_{year}.json", {p:dict(n=len(ix), start=str(pd.Timestamp(a['dates'][ix[0]]).date()),
                    end=str(pd.Timestamp(a['dates'][ix[-1]]).date())) for p,ix in split.items()})
                prior, masks, truth = {}, {}, {}
                for task in groups:
                    y = a[task+"_raw_relative_target"]
                    p, _, _, available, count = fit_prior(y[ti], a[task+"_target_mask"][ti], groups[task])
                    prior[task] = p
                    truth[task] = y
                    masks[task] = a[task+"_target_mask"].astype(bool) & available[None] & (count[None] >= old["minimum_train_days"])
                baselines = {"prior":{t:np.broadcast_to(prior[t], truth[t].shape) for t in groups}}
                for name in ["momentum20", "cohort", "price_soft"]:
                    x = signals[name]
                    weights = a["sleeve_reference_weights"]
                    sw = np.where(np.isfinite(x[:, None, :]), x[:, None, :], 0)
                    sleeve = np.sum(weights*sw, -1)
                    valid = (weights.sum(-1)>0) & ((weights == 0) | np.isfinite(x[:, None, :])).all(-1)
                    sleeve = np.where(valid, sleeve, np.nan)
                    baselines[name] = dict(asset=np.repeat(x[:, None, :], 3, 1), sleeve=np.repeat(sleeve[:, None, :], 3, 1))
                # Only as-of baseline availability constrains common comparison support.
                common = {t:masks[t].copy() for t in groups}
                for base in baselines.values():
                    for t in groups:
                        common[t] &= np.isfinite(base[t])
                for part,ix in split.items():
                    for t in groups:
                        rows.append(dict(year=year, part=part, task=t, dates=len(ix),
                            valid_cells=int(common[t][ix].sum()), entities_with_any=int(common[t][ix].any((0, 1)).sum())))
                vy, vm = {t:truth[t][vi] for t in groups}, {t:common[t][vi] for t in groups}
                vb = {name:{t:v[vi] for t,v in base.items()} for name,base in baselines.items()}
                references[year] = dict(truth=vy, mask=vm, baselines=vb)
                cy, cm = {t:truth[t][ci] for t in groups}, {t:common[t][ci] for t in groups}
                for seed in cfg["seeds"]:
                    folder = models / str(year) / str(seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    identity = dict(manifest=manifest, year=year, seed=seed, initialized_from="random_inner_T_only")
                    done = folder / "summary.json"
                    if done.exists():
                        summary = json.loads(done.read_text())
                        if summary["identity"] != identity:
                            raise ValueError("Completed ranker identity mismatch")
                    else:
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        model = StateRanker(engine.ForecastNetwork(clipped, old), old["d_model"]).to(device)
                        optimizer = torch.optim.AdamW([dict(params=model.heads.parameters(), lr=cfg["head_lr"]),
                            dict(params=model.network.core.parameters(), lr=cfg["encoder_lr"])], weight_decay=cfg["weight_decay"])
                        initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        rng = np.random.default_rng(seed)
                        best, best_epoch, first = -float("inf"), 0, 1
                        best_state, logs = copy.deepcopy(model.state_dict()), []
                        current = folder / "current.pt"
                        if current.exists():
                            saved = torch.load(current, map_location="cpu", weights_only=False)
                            assert saved["identity"] == identity
                            model.load_state_dict(saved["state"])
                            optimizer.load_state_dict(saved["optimizer"])
                            best, best_epoch, first = saved["best"], saved["best_epoch"], saved["epoch"]+1
                            best_state, logs = saved["best_state"], saved["logs"]
                            rng.bit_generator.state = saved["numpy_rng"]
                            torch.set_rng_state(saved["torch_rng"])
                            torch.cuda.set_rng_state_all(saved["cuda_rng"])
                        def evaluate():
                            cscore = infer(model, clipped, ci, old, device, cfg["batch_size"])
                            vscore = infer(model, clipped, vi, old, device, cfg["batch_size"])
                            beta = {t:fit_calibrator(cscore[t], cy[t], cm[t], prior[t], groups[t], cfg["calibration_ridge"]) for t in groups}
                            raw = {t:prior[t]+beta[t]*vscore[t] for t in groups}
                            metric, delta, ic = rank_metrics(vscore, vb, vy, vm, groups)
                            mse, _ = score(raw, vb["prior"], vy, vm, groups)
                            return metric, delta, ic, mse, vscore, raw, beta
                        for epoch in range(first, cfg["maximum_epochs"]+1):
                            model.train()
                            order = rng.permutation(len(ti))
                            losses = []
                            for start in range(0, len(order), cfg["batch_size"]):
                                ix = ti[order[start:start+cfg["batch_size"]]]
                                batch = engine.batch_at(clipped, ix, old, device)
                                y = {t:torch.tensor(truth[t][ix], device=device) for t in groups}
                                m = {t:torch.tensor(masks[t][ix], device=device) for t in groups}
                                optimizer.zero_grad(set_to_none=True)
                                pred = model(batch)
                                loss = rank_loss(pred, y, m, gt, cfg["score_penalty"])
                                if not torch.isfinite(loss):
                                    raise FloatingPointError("Nonfinite rank loss")
                                loss.backward()
                                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                                optimizer.step()
                                losses.append(float(loss.detach()))
                            if epoch == 1 or epoch % cfg["validation_every"] == 0:
                                metric, delta, ic, mse, _, _, _ = evaluate()
                                if not np.isfinite(metric):
                                    raise ValueError("Insufficient common validation coverage")
                                if metric > best+1e-4:
                                    best, best_epoch, best_state = metric, epoch, copy.deepcopy(model.state_dict())
                                row = dict(year=year, seed=seed, epoch=epoch, train_loss=float(np.mean(losses)),
                                    validation_min_rank_increment=metric, validation_rank_ic=ic,
                                    validation_calibrated_MSE_ratio_prior=mse, best_epoch=best_epoch, gradient_norm=float(gradient))
                                logs.append(row)
                                pd.DataFrame(logs).to_csv(folder / "loss.csv", index=False)
                                progress["current"] = row
                                save(out / "progress.json", progress)
                                print(json.dumps(row), flush=True)
                            atomic_checkpoint(dict(identity=identity, epoch=epoch, state=model.state_dict(),
                                optimizer=optimizer.state_dict(), best=best, best_epoch=best_epoch,
                                best_state=best_state, logs=logs, numpy_rng=rng.bit_generator.state,
                                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), current)
                            if epoch >= cfg["minimum_epochs"] and epoch-best_epoch >= cfg["patience_epochs"]:
                                break
                        model.load_state_dict(best_state)
                        metric, delta, ic, mse, scores, raw, beta = evaluate()
                        changed = [k for k,v in model.state_dict().items() if not torch.equal(v.cpu(), initial[k])]
                        core_changed = [k for k in changed if k.startswith("network.core.")]
                        if not core_changed or any(not k.startswith(("network.core.", "heads.")) for k in changed):
                            raise ValueError("End-to-end update boundary not satisfied")
                        summary = dict(identity=identity, year=year, seed=seed, selected_epoch=best_epoch, trained_epochs=epoch,
                            validation_min_rank_increment=metric, validation_delta_by_baseline=delta,
                            validation_rank_ic=ic, validation_calibrated_MSE_ratio_prior=mse,
                            encoder_updated_parameters=core_changed, validation_not_tradeable=True,
                            score_units="dimensionless_relative_rank", mean_units="cumulative_raw_relative_price_return",
                            formal_promotion=False)
                        atomic_checkpoint(dict(identity=identity, state=model.state_dict(), calibrator=beta, prior=prior), folder / "selected.pt")
                        np.savez_compressed(folder / "validation_predictions.npz", **{t:scores[t] for t in groups}, **{t+"_mean":raw[t] for t in groups})
                        save(done, summary)
                        del model, optimizer
                        torch.cuda.empty_cache()
                    with np.load(folder / "validation_predictions.npz") as z:
                        predictions.setdefault(year, []).append({k:z[k].copy() for k in z.files})
                    summaries.append(summary)
                    progress["completed"] = len(summaries)
                    save(out / "progress.json", progress)
        checks = dict(all_seeds_rank_nonworse=all(s["validation_min_rank_increment"] >= 0 for s in summaries),
            calibrated_MSE_nonworse=all(s["validation_calibrated_MSE_ratio_prior"] <= 1+1e-7 for s in summaries))
        details = []
        for year in cfg["years"]:
            r, values = references[year], predictions[year]
            pred = {t:np.mean([v[t] for v in values], 0) for t in groups}
            metric, delta, ic = rank_metrics(pred, r["baselines"], r["truth"], r["mask"], groups)
            checks[f"{year}_ensemble_rank_increment"] = metric >= cfg["minimum_rank_increment"]
            agree = {}
            for task in groups:
                correlations = []
                for left in range(3):
                    for right in range(left+1, 3):
                        for h in range(3):
                            for g in np.unique(groups[task]):
                                members = groups[task] == g
                                correlations.append(engine.finite_mean(engine.daily_ic(values[left][task][:, h, members],
                                    values[right][task][:, h, members], r["mask"][task][:, h, members])))
                agree[task] = engine.finite_mean(correlations)
                checks[f"{year}_{task}_seed_agreement"] = agree[task] >= (.5 if task == "asset" else .65)
            details.append(dict(year=year, min_rank_increment=metric, delta_by_baseline=delta, rank_ic=ic, seed_agreement=agree))
        decision = dict(checks=checks, research_followup_allowed=all(checks.values()), details=details,
            formal_promotion=False, ppo_training_started=False, boundary_enabled=False, G_or_test_used=False,
            next_step="freeze separate followup" if all(checks.values()) else "stop; no automatic search")
        save(out / "decision.json", decision)
        pd.DataFrame(rows).to_csv(out / "common_support.csv", index=False)
        pd.DataFrame([{k:v for k,v in s.items() if k not in ["identity", "encoder_updated_parameters"]} for s in summaries]).to_csv(out / "training_summary.csv", index=False)
        (out / "experiment_review.md").write_text("# V61 条件动量排序实验\n\n6组三种子两折完整监督训练完成。\n\n"
            +f"下一阶段研究资格：{decision['research_followup_allowed']}。\n\n"
            +"失败项："+"、".join(k for k,v in checks.items() if not v)+"\n\n"
            +"价值Boundary仍缺核验，价格过热修正不冒充价值边界；原版和九宫格保留。C用于校准收益单位，V用于选择，因此V预测非当时可交易观点，不得回填历史。正式EP/PPO和新夏普未获授权。详见decision.json、common_support.csv及逐组summary.json。\n", encoding="utf-8")
        progress.update(status="completed_validation_only", research_followup_allowed=decision["research_followup_allowed"])
        save(out / "progress.json", progress)
    except BaseException as exc:
        progress.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(out / "progress.json", progress)
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    main()
