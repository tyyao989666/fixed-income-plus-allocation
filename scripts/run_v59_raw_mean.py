"""Two preregistered heads; historical T/V only; no G/test or portfolio training."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation
from src.cuda_finite_check import noncompacting_finite_check
from src.windows_checkpoint_io import atomic_checkpoint
from src.v59_raw_mean import RawMeanHead, fit_shared_scale, raw_mean_loss
from scripts.run_v45_research_pipeline import process_alive


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    engine.write_json(path, obj)


def score(predictions, bases, targets, masks, groups):
    ratios, deltas = [], []
    for task in predictions:
        for h in range(3):
            terms, ranks = [], []
            for group in np.unique(groups[task]):
                members = groups[task] == group
                y, p, b = [v[task][:, h, members] for v in [targets, predictions, bases]]
                m = masks[task][:, h, members] & np.isfinite(b) & np.isfinite(y)
                if not np.isfinite(p[m]).all():
                    raise FloatingPointError("Nonfinite validation mean")
                n = m.sum(1)
                valid = n > 0
                if not valid.any():
                    continue
                mse = lambda value: np.mean(((np.where(m, value-y, 0)**2).sum(1)/np.maximum(n, 1))[valid])
                base_mse = mse(b)
                if base_mse <= 1e-16:
                    continue
                terms.append(mse(p)/base_mse)
                ranks.append(engine.finite_mean(engine.daily_ic(p, y, m)-engine.daily_ic(b, y, m)))
            if terms:
                ratios.append(float(np.mean(terms)))
                deltas.append(engine.finite_mean(ranks))
    if not ratios:
        raise ValueError("No usable matched validation cells")
    return float(np.mean(ratios)), engine.finite_mean(deltas)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU interpreter required")
    cfg = json.loads((ROOT / "config/v59_raw_mean.json").read_text())
    report = ROOT / "output/reports/raw_mean_v59"
    models = ROOT / "output/models/raw_mean_v59"
    report.mkdir(parents=True, exist_ok=True)
    lock = report / "active.json"
    if lock.exists() and process_alive(json.loads(lock.read_text())["pid"]):
        raise RuntimeError("V59 already active")
    source_paths = [Path(__file__), ROOT / "src/v59_raw_mean.py", ROOT / "config/v59_raw_mean.json",
        ROOT / "docs/v59_raw_mean_preregistration.md", ROOT / "src/cuda_finite_check.py",
        ROOT / "src/v45_forecast.py", ROOT / "src/v49_masked_forecast.py", ROOT / "src/windows_checkpoint_io.py"]
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    dataset = ROOT / "data_processed/corporate_action_v58/dataset.npz"
    registered = json.loads((ROOT / "output/reports/corporate_action_v58/status.json").read_text())
    assert sha(dataset) == registered["dataset_sha256"]
    provenance["dataset_sha256"] = sha(dataset)
    for year in cfg["years"]:
        for seed in cfg["seeds"]:
            for p in [ROOT / f"output/models/corporate_action_v58/{year}/seed_{seed}_pretraining.pt",
                      ROOT / f"output/models/corporate_action_v58/{year}/seed_{seed}_F1.pt",
                      ROOT / f"output/models/multitask_v58/{year}/{seed}/forecast_arrays.npz"]:
                provenance[str(p.relative_to(ROOT))] = sha(p)
    parent_meta = json.loads((ROOT / "output/reports/corporate_action_v58/metadata.json").read_text())
    for source in parent_meta["source_manifest"].values():
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Parent source lineage changed")
    frozen = report / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != provenance:
        raise ValueError("Implementation changed; refuse incompatible resume")
    save(frozen, provenance)
    with np.load(dataset) as z:
        a = {k: z[k].copy() for k in z.files}
    assert len(a["module_index"]) == 38
    oldcfg = engine.configuration(ROOT / "config/v45_research.yaml")
    oldcfg["seed_workers"] = 1
    torch.set_num_threads(2)
    device = torch.device("cuda")
    groups = {"asset": a["module_index"], "sleeve": np.zeros(5, int)}
    group_tensors = {k: torch.tensor(v, device=device) for k, v in groups.items()}
    state = dict(status="training", pid=os.getpid(), completed=0, total=12,
                 formal_promotion=False, ppo_training_started=False, evaluation_parts=["train", "validation"])
    save(lock, {"pid": os.getpid()})
    save(report / "progress.json", state)
    summaries, ensemble_data = [], {}
    try:
        with training_implementation(), noncompacting_finite_check():
            for year in cfg["years"]:
                split = engine.shared_split(a, year, oldcfg)
                assert split["valid"]
                clipped, _ = engine.clip_fold_features(a, split["train"], oldcfg)
                centers, scales, masks, targets = {}, {}, {}, {}
                for task in groups:
                    c, s, ok = fit_shared_scale(a[task+"_raw_relative_target"][split["train"]],
                        a[task+"_target_mask"][split["train"]], groups[task])
                    centers[task], scales[task] = c, s
                    targets[task] = a[task+"_raw_relative_target"]
                    masks[task] = a[task+"_target_mask"].astype(bool) & ok[None]
                baseline = torch.load(ROOT / f"output/models/corporate_action_v58/{year}/seed_17_F1.pt", map_location="cpu", weights_only=False)["baseline"]
                vi = split["validation"]
                b = baseline.predict(engine.raw_features(a, vi))
                bases = {t: (b[t]+a[t+"_opportunity_offset"][vi])*a[t+"_opportunity_scale"][vi] for t in groups}
                # One shared mask for both candidates and F0; no future-label-based filling.
                vm = {t: masks[t][vi] & np.isfinite(bases[t]) for t in groups}
                vy = {t: targets[t][vi] for t in groups}
                for seed in cfg["seeds"]:
                    source = ROOT / f"output/models/corporate_action_v58/{year}/seed_{seed}_pretraining.pt"
                    identity = dict(provenance=provenance, encoder_sha256=sha(source), year=year, seed=seed)
                    cache = models / str(year) / str(seed)
                    cache.mkdir(parents=True, exist_ok=True)
                    cachefile = cache / "historical_embeddings.pt"
                    if cachefile.exists():
                        stored = torch.load(cachefile, map_location="cpu", weights_only=False)
                        if stored["identity"] != identity:
                            raise ValueError("Embedding cache identity mismatch")
                        features = stored["features"]
                    else:
                        encoder = engine.ForecastNetwork(clipped, oldcfg).to(device)
                        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
                        assert checkpoint["complete"]
                        encoder.load_state_dict(checkpoint["best_state"])
                        features = {part: engine.infer(encoder, clipped, split[part], oldcfg, device, True)
                                    for part in ["train", "validation"]}
                        atomic_checkpoint(dict(identity=identity, features=features), cachefile)
                        del encoder, checkpoint
                    ft = {part: {t: torch.tensor(v, device=device) for t, v in f.items()} for part, f in features.items()}
                    train_y, train_m = {}, {}
                    for task in groups:
                        ix = split["train"]
                        train_y[task] = torch.tensor(np.where(masks[task][ix], (targets[task][ix]-centers[task])/scales[task], 0), device=device)
                        train_m[task] = torch.tensor(masks[task][ix], device=device)
                    for kind in cfg["candidates"]:
                        folder = cache / kind
                        folder.mkdir(exist_ok=True)
                        ident = dict(identity, candidate=kind)
                        completed = folder / "summary.json"
                        if completed.exists():
                            summary = json.loads(completed.read_text())
                            if summary["identity"] != ident:
                                raise ValueError("Completed candidate identity mismatch")
                        else:
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                            heads = nn.ModuleDict({t: RawMeanHead(oldcfg["d_model"], kind, cfg["dropout"]) for t in groups}).to(device)
                            opt = torch.optim.AdamW(heads.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
                            rng = np.random.default_rng(seed)
                            best, best_epoch, start = float("inf"), 0, 1
                            best_state, log = copy.deepcopy(heads.state_dict()), []
                            resume_path = folder / "current.pt"
                            if resume_path.exists():
                                saved = torch.load(resume_path, map_location="cpu", weights_only=False)
                                assert saved["identity"] == ident
                                heads.load_state_dict(saved["state"])
                                opt.load_state_dict(saved["optimizer"])
                                best, best_epoch, start = saved["best"], saved["best_epoch"], saved["epoch"]+1
                                best_state, log = saved["best_state"], saved["log"]
                                rng.bit_generator.state = saved["numpy_rng"]
                                torch.set_rng_state(saved["torch_rng"])
                                torch.cuda.set_rng_state_all(saved["cuda_rng"])
                            def predict():
                                heads.eval()
                                with torch.no_grad():
                                    return {t: heads[t](ft["validation"][t])[0].cpu().numpy()*scales[t]+centers[t] for t in groups}
                            for epoch in range(start, cfg["maximum_epochs"]+1):
                                heads.train()
                                losses = []
                                order = rng.permutation(len(split["train"]))
                                for begin in range(0, len(order), cfg["batch_size"]):
                                    ix = order[begin:begin+cfg["batch_size"]]
                                    opt.zero_grad(set_to_none=True)
                                    loss = sum(.5*raw_mean_loss(*heads[t](ft["train"][t][ix]), train_y[t][ix], train_m[t][ix], group_tensors[t], cfg) for t in groups)
                                    if not torch.isfinite(loss):
                                        raise FloatingPointError("Nonfinite raw objective")
                                    loss.backward()
                                    nn.utils.clip_grad_norm_(heads.parameters(), 1., error_if_nonfinite=True)
                                    opt.step()
                                    losses.append(float(loss.detach()))
                                if epoch == 1 or epoch % cfg["validation_every"] == 0:
                                    ratio, delta = score(predict(), bases, vy, vm, groups)
                                    if ratio < best-1e-4:
                                        best, best_epoch, best_state = ratio, epoch, copy.deepcopy(heads.state_dict())
                                    row = dict(year=year, seed=seed, candidate=kind, epoch=epoch, train_loss=float(np.mean(losses)), validation_MSE_ratio_F0=ratio, validation_raw_delta_rank_ic=delta, best_epoch=best_epoch)
                                    log.append(row)
                                    pd.DataFrame(log).to_csv(folder / "loss.csv", index=False)
                                    state.update(current=row)
                                    save(report / "progress.json", state)
                                    print(json.dumps(row), flush=True)
                                atomic_checkpoint(dict(identity=ident, state=heads.state_dict(), optimizer=opt.state_dict(), best_state=best_state,
                                    best=best, best_epoch=best_epoch, epoch=epoch, log=log, numpy_rng=rng.bit_generator.state,
                                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), resume_path)
                                if epoch >= cfg["minimum_epochs"] and epoch-best_epoch >= cfg["patience_epochs"]:
                                    break
                            heads.load_state_dict(best_state)
                            pred = predict()
                            ratio, delta = score(pred, bases, vy, vm, groups)
                            old_bundle = ROOT / f"output/models/multitask_v58/{year}/{seed}/forecast_arrays.npz"
                            with np.load(old_bundle) as old:
                                old_raw = {t: (old["validation_"+t]+a[t+"_opportunity_offset"][vi])*a[t+"_opportunity_scale"][vi] for t in groups}
                            old_ratio, old_delta = score(old_raw, bases, vy, vm, groups)
                            coverage = {}
                            quantiles = {}
                            with torch.no_grad():
                                for t in groups:
                                    _, q = heads[t](ft["validation"][t])
                                    quantiles[t] = q.cpu().numpy()*scales[t][None, ..., None]+centers[t][None, ..., None]
                                    q = quantiles[t]
                                    coverage[t] = float(((vy[t] >= q[..., 0]) & (vy[t] <= q[..., 2]))[vm[t]].mean()) if vm[t].any() else None
                            atomic_checkpoint(dict(identity=ident, state=best_state, centers=centers, scales=scales), folder / "selected.pt")
                            np.savez_compressed(folder / "validation_predictions.npz", **pred, **{t+"_quantiles":q for t,q in quantiles.items()})
                            summary = dict(identity=ident, year=year, seed=seed, candidate=kind, best_epoch=best_epoch,
                                trained_epochs=epoch, validation_MSE_ratio_F0=ratio, validation_improvement=1-ratio,
                                validation_raw_delta_rank_ic=delta, V58_reference_MSE_ratio_F0=old_ratio,
                                V58_reference_raw_delta_rank_ic=old_delta, interval_coverage_cell_weighted=coverage,
                                raw_units="cumulative_raw_relative_price_return", formal_promotion=False)
                            save(completed, summary)
                            del heads, opt
                        summaries.append(summary)
                        with np.load(folder / "validation_predictions.npz") as z:
                            pred = {t: z[t].copy() for t in groups}
                        ensemble_data.setdefault((year, kind), []).append(pred)
                        state["completed"] = len(summaries)
                        save(report / "progress.json", state)
                    del ft, train_y, train_m
                    torch.cuda.empty_cache()
        results = []
        for kind in cfg["candidates"]:
            selected = [s for s in summaries if s["candidate"] == kind]
            checks = dict(all_seed_MSE_nonworse=min(s["validation_improvement"] for s in selected) >= cfg["minimum_seed_validation_improvement"],
                          mean_MSE_improvement=float(np.mean([s["validation_improvement"] for s in selected])) >= cfg["minimum_mean_validation_improvement"])
            details = []
            for year in cfg["years"]:
                vi = engine.shared_split(a, year, oldcfg)["validation"]
                baseline = torch.load(ROOT / f"output/models/corporate_action_v58/{year}/seed_17_F1.pt", map_location="cpu", weights_only=False)["baseline"]
                bb = baseline.predict(engine.raw_features(a, vi))
                base = {t: (bb[t]+a[t+"_opportunity_offset"][vi])*a[t+"_opportunity_scale"][vi] for t in groups}
                preds = ensemble_data[year, kind]
                pred = engine.ensemble(preds)
                truth = {t: a[t+"_raw_relative_target"][vi] for t in groups}
                mask = {t: a[t+"_target_mask"][vi].astype(bool) & np.isfinite(base[t]) for t in groups}
                ratio, delta = score(pred, base, truth, mask, groups)
                checks[f"{year}_ensemble_rank_nonworse"] = delta >= 0
                agreements = {}
                for task in groups:
                    vals = []
                    for i in range(3):
                        for j in range(i+1, 3):
                            for h in range(3):
                                for group in np.unique(groups[task]):
                                    members = groups[task] == group
                                    vals.append(engine.finite_mean(engine.daily_ic(preds[i][task][:, h, members], preds[j][task][:, h, members], mask[task][:, h, members])))
                    agreements[task] = engine.finite_mean(vals)
                    checks[f"{year}_{task}_seed_agreement"] = agreements[task] >= (.5 if task == "asset" else .65)
                details.append(dict(year=year, MSE_ratio_F0=ratio, raw_delta_rank_ic=delta, seed_agreement=agreements))
            results.append(dict(candidate=kind, checks=checks, research_followup_allowed=all(checks.values()), details=details))
        allowed = [r["candidate"] for r in results if r["research_followup_allowed"]]
        save(report / "decision.json", dict(candidates=results, selected_for_followup=allowed[0] if allowed else None,
             formal_promotion=False, ppo_training_started=False, test_labels_used=False, next_step="freeze separate followup" if allowed else "stop; no threshold relaxation or automatic search"))
        pd.DataFrame([{k:v for k,v in s.items() if k != "identity"} for s in summaries]).to_csv(report / "training_summary.csv", index=False)
        state.update(status="completed_validation_only", formal_promotion=False)
        save(report / "progress.json", state)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(report / "progress.json", state)
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    main()
