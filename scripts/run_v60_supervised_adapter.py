"""Paired historical T/V experiment: frozen vs supervised final iTransformer block."""
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
from src.v60_supervised_adapter import SupervisedAdapter, capture_prefix, fit_prior, supervised_loss
from src.windows_checkpoint_io import atomic_checkpoint
from scripts.run_v59_raw_mean import sha, save, score
from scripts.run_v45_research_pipeline import process_alive


@torch.no_grad()
def predict(model, prefix, batch_size):
    model.eval()
    rows = {t:[] for t in ["asset", "sleeve"]}
    for start in range(0, len(prefix["tokens"]), batch_size):
        values = {k:v[start:start+batch_size] for k,v in prefix.items()}
        pred, _ = model(values)
        for t,v in model.decode(pred).items():
            rows[t].append(v.cpu().numpy())
    return {t:np.concatenate(v) for t,v in rows.items()}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA interpreter required")
    cfg = json.loads((ROOT / "config/v60_supervised_adapter.json").read_text())
    out = ROOT / "output/reports/supervised_adapter_v60"
    models = ROOT / "output/models/supervised_adapter_v60"
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "active.json"
    if lock.exists() and process_alive(json.loads(lock.read_text())["pid"]):
        raise RuntimeError("V60 already active")
    files = [Path(__file__), ROOT / "src/v60_supervised_adapter.py", ROOT / "src/v59_raw_mean.py",
        ROOT / "scripts/run_v59_raw_mean.py", ROOT / "config/v60_supervised_adapter.json",
        ROOT / "docs/v60_supervised_adapter_preregistration.md", ROOT / "src/v45_forecast.py",
        ROOT / "src/v49_masked_forecast.py", ROOT / "src/cuda_finite_check.py", ROOT / "src/windows_checkpoint_io.py",
        ROOT / "config/v45_research.yaml", ROOT / "src/hierarchical_asset_itransformer.py"]
    for year in cfg["years"]:
        files.append(ROOT / f"output/models/corporate_action_v58/{year}/seed_{cfg['encoder_seed']}_pretraining.pt")
    manifest = {str(p.relative_to(ROOT)):sha(p) for p in files}
    data = ROOT / "data_processed/corporate_action_v58/dataset.npz"
    manifest["dataset_sha256"] = sha(data)
    registered = json.loads((ROOT / "output/reports/corporate_action_v58/status.json").read_text())
    assert manifest["dataset_sha256"] == registered["dataset_sha256"]
    meta = json.loads((ROOT / "output/reports/corporate_action_v58/metadata.json").read_text())
    for source in meta["source_manifest"].values():
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Source lineage changed")
    frozen = out / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise ValueError("V60 manifest changed; refusing incompatible resume")
    save(frozen, manifest)
    with np.load(data) as z:
        a = {k:z[k].copy() for k in z.files}
    assert len(a["module_index"]) == 38
    old = engine.configuration(ROOT / "config/v45_research.yaml")
    old["seed_workers"] = 1
    torch.set_num_threads(2)
    # A capture hook disables the fused encoder fastpath on the source layer.
    # Use the same unfused path for source and replay, rather than relax parity.
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda")
    groups = dict(asset=a["module_index"], sleeve=np.zeros(5, int))
    gt = {t:torch.tensor(g, device=device) for t,g in groups.items()}
    progress = dict(status="training", pid=os.getpid(), completed=0, total=12,
        formal_promotion=False, ppo_training_started=False, evaluation_parts=["train", "validation"],
        encoder_seed=cfg["encoder_seed"])
    save(lock, dict(pid=os.getpid()))
    save(out / "progress.json", progress)
    summaries, predictions, references = [], {}, {}
    try:
        with training_implementation(), noncompacting_finite_check():
            for year in cfg["years"]:
                split = engine.shared_split(a, year, old)
                assert split["valid"]
                clipped, _ = engine.clip_fold_features(a, split["train"], old)
                source = ROOT / f"output/models/corporate_action_v58/{year}/seed_{cfg['encoder_seed']}_pretraining.pt"
                network = engine.ForecastNetwork(clipped, old).to(device)
                loaded = torch.load(source, map_location="cpu", weights_only=False)
                assert loaded["complete"]
                network.load_state_dict(loaded["best_state"])
                cache = models / str(year)
                cache.mkdir(parents=True, exist_ok=True)
                cached = cache / "prefix.pt"
                if cached.exists():
                    payload = torch.load(cached, map_location="cpu", weights_only=False)
                    if payload["manifest"] != manifest:
                        raise ValueError("Prefix lineage mismatch")
                    prefixes = payload["prefixes"]
                else:
                    prefixes = {}
                    for part in ["train", "validation"]:
                        chunks = []
                        for start in range(0, len(split[part]), old["pretrain_batch"]):
                            ix = split[part][start:start+old["pretrain_batch"]]
                            batch = engine.batch_at(clipped, ix, old, device)
                            prefix, expected = capture_prefix(network, batch)
                            dummy = {t:np.zeros((3, 38 if t == "asset" else 5), np.float32) for t in groups}
                            unit = {t:np.ones_like(v) for t,v in dummy.items()}
                            replay = SupervisedAdapter(network.core, dummy, dummy, unit, "frozen_control").to(device).eval()
                            with torch.no_grad():
                                actual = replay.representations(prefix)
                            for t in groups:
                                torch.testing.assert_close(actual[t], expected[t], atol=1e-5, rtol=1e-5)
                            chunks.append({k:v.cpu() for k,v in prefix.items()})
                            del replay
                        prefixes[part] = {k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}
                    atomic_checkpoint(dict(manifest=manifest, prefixes=prefixes, replay_parity_passed=True), cached)
                prefix_gpu = {p:{k:v.to(device) for k,v in values.items()} for p,values in prefixes.items()}
                priors, centers, scales, masks, target = {}, {}, {}, {}, {}
                for t in groups:
                    prior, center, scale, available, count = fit_prior(a[t+"_raw_relative_target"][split["train"]],
                        a[t+"_target_mask"][split["train"]], groups[t], cfg["prior_shrinkage_days"])
                    priors[t], centers[t], scales[t] = prior, center, scale
                    masks[t] = a[t+"_target_mask"].astype(bool) & available[None]
                    target[t] = a[t+"_raw_relative_target"]
                vi, ti = split["validation"], split["train"]
                base = {t:np.broadcast_to(priors[t], target[t][vi].shape) for t in groups}
                vy, vm = {t:target[t][vi] for t in groups}, {t:masks[t][vi] for t in groups}
                references[year] = dict(base=base, truth=vy, mask=vm)
                ty = {t:torch.tensor(np.where(masks[t][ti], (target[t][ti]-centers[t])/scales[t], 0), device=device) for t in groups}
                tm = {t:torch.tensor(masks[t][ti], device=device) for t in groups}
                for seed in cfg["training_seeds"]:
                    for mode in cfg["modes"]:
                        folder = cache / str(seed) / mode
                        folder.mkdir(parents=True, exist_ok=True)
                        identity = dict(manifest=manifest, year=year, training_seed=seed, mode=mode)
                        done = folder / "summary.json"
                        if done.exists():
                            summary = json.loads(done.read_text())
                            if summary["identity"] != identity:
                                raise ValueError("Completed task lineage mismatch")
                        else:
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                            model = SupervisedAdapter(network.core, priors, centers, scales, mode,
                                cfg["residual_bound_shared_sigma"]).to(device)
                            initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            params = [dict(params=model.heads.parameters(), lr=cfg["head_lr"])]
                            if mode == "last_block_adapter":
                                params.append(dict(params=model.core.encoder.layers[-1].parameters(), lr=cfg["encoder_lr"]))
                            optimizer = torch.optim.AdamW(params, weight_decay=cfg["weight_decay"])
                            best, best_epoch, first = 1., 0, 1
                            best_state, log = copy.deepcopy(model.state_dict()), []
                            rng = np.random.default_rng(seed)
                            current = folder / "current.pt"
                            if current.exists():
                                saved = torch.load(current, map_location="cpu", weights_only=False)
                                assert saved["identity"] == identity
                                model.load_state_dict(saved["state"])
                                optimizer.load_state_dict(saved["optimizer"])
                                best, best_epoch, first = saved["best"], saved["best_epoch"], saved["epoch"]+1
                                best_state, log = saved["best_state"], saved["log"]
                                rng.bit_generator.state = saved["numpy_rng"]
                                torch.set_rng_state(saved["torch_rng"])
                                torch.cuda.set_rng_state_all(saved["cuda_rng"])
                            for epoch in range(first, cfg["maximum_epochs"]+1):
                                model.train()
                                order = rng.permutation(len(ti))
                                losses = []
                                for start in range(0, len(order), cfg["batch_size"]):
                                    ix = order[start:start+cfg["batch_size"]]
                                    optimizer.zero_grad(set_to_none=True)
                                    p, d = model({k:v[ix] for k,v in prefix_gpu["train"].items()})
                                    loss = supervised_loss(p, d, {t:v[ix] for t,v in ty.items()}, {t:v[ix] for t,v in tm.items()}, gt, cfg)
                                    if not torch.isfinite(loss):
                                        raise FloatingPointError("Nonfinite supervised loss")
                                    loss.backward()
                                    nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                                    optimizer.step()
                                    losses.append(float(loss.detach()))
                                if epoch == 1 or epoch % cfg["validation_every"] == 0:
                                    pred = predict(model, prefix_gpu["validation"], cfg["batch_size"])
                                    ratio, delta = score(pred, base, vy, vm, groups)
                                    if ratio < best-1e-4:
                                        best, best_epoch, best_state = ratio, epoch, copy.deepcopy(model.state_dict())
                                    row = dict(year=year, seed=seed, mode=mode, epoch=epoch,
                                        train_loss=float(np.mean(losses)), validation_MSE_ratio_prior=ratio,
                                        raw_delta_rank_ic=delta, best_epoch=best_epoch, gradient_norm=float(nnorm))
                                    log.append(row)
                                    pd.DataFrame(log).to_csv(folder / "loss.csv", index=False)
                                    progress["current"] = row
                                    save(out / "progress.json", progress)
                                    print(json.dumps(row), flush=True)
                                atomic_checkpoint(dict(identity=identity, state=model.state_dict(), optimizer=optimizer.state_dict(),
                                    best=best, best_epoch=best_epoch, best_state=best_state, epoch=epoch, log=log,
                                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), current)
                                if epoch >= cfg["minimum_epochs"] and epoch-best_epoch >= cfg["patience_epochs"]:
                                    break
                            model.load_state_dict(best_state)
                            pred = predict(model, prefix_gpu["validation"], cfg["batch_size"])
                            ratio, delta = score(pred, base, vy, vm, groups)
                            changed = [k for k,v in model.state_dict().items() if not torch.equal(v.cpu(), initial[k])]
                            last = len(model.core.encoder.layers)-1
                            allowed = ("heads.", f"core.encoder.layers.{last}.") if mode == "last_block_adapter" else ("heads.",)
                            if any(not k.startswith(allowed) for k in changed):
                                raise ValueError("Frozen ownership boundary changed")
                            summary = dict(identity=identity, year=year, seed=seed, mode=mode, selected_epoch=best_epoch,
                                trained_epochs=epoch, validation_MSE_ratio_prior=ratio, improvement_vs_prior=1-ratio,
                                raw_delta_rank_ic=delta, changed_parameters=changed,
                                last_block_changed=any(k.startswith(f"core.encoder.layers.{last}.") for k in changed),
                                frozen_boundary_passed=True, formal_promotion=False, validation_not_tradeable=True)
                            atomic_checkpoint(dict(identity=identity, state=model.state_dict()), folder / "selected.pt")
                            np.savez_compressed(folder / "validation_predictions.npz", **pred)
                            save(done, summary)
                            del model, optimizer
                        with np.load(folder / "validation_predictions.npz") as z:
                            predictions.setdefault((year, mode), []).append({t:z[t].copy() for t in groups})
                        summaries.append(summary)
                        progress["completed"] = len(summaries)
                        save(out / "progress.json", progress)
                del network, loaded, prefix_gpu, ty, tm
                torch.cuda.empty_cache()
        table = pd.DataFrame([{k:v for k,v in s.items() if k not in ["identity", "changed_parameters"]} for s in summaries])
        table.to_csv(out / "training_summary.csv", index=False)
        adapted = table[table['mode'].eq("last_block_adapter")]
        checks = dict(all_adapter_runs_nonworse=bool((adapted.improvement_vs_prior >= 0).all()),
            mean_improvement=bool(adapted.improvement_vs_prior.mean() >= cfg["minimum_mean_MSE_improvement"]))
        details = []
        for year in cfg["years"]:
            reference = references[year]
            modes = {}
            for mode in cfg["modes"]:
                values = predictions[year, mode]
                ratio, delta = score(engine.ensemble(values), reference["base"], reference["truth"], reference["mask"], groups)
                agree = {}
                for task in groups:
                    terms = []
                    for left in range(3):
                        for right in range(left+1, 3):
                            for h in range(3):
                                for group in np.unique(groups[task]):
                                    members = groups[task] == group
                                    terms.append(engine.finite_mean(engine.daily_ic(values[left][task][:, h, members],
                                        values[right][task][:, h, members], reference["mask"][task][:, h, members])))
                    agree[task] = engine.finite_mean(terms)
                modes[mode] = dict(MSE_ratio_prior=ratio, raw_delta_rank_ic=delta, seed_agreement=agree)
            adapt, control = modes["last_block_adapter"], modes["frozen_control"]
            checks[f"{year}_rank_nonworse"] = adapt["raw_delta_rank_ic"] >= 0
            checks[f"{year}_paired_nonworse"] = adapt["MSE_ratio_prior"] <= control["MSE_ratio_prior"]
            for t in groups:
                checks[f"{year}_{t}_seed_agreement"] = adapt["seed_agreement"][t] >= (.5 if t == "asset" else .65)
            details.append(dict(year=year, modes=modes))
        control_runs = table[table['mode'].eq("frozen_control")]
        checks["paired_overall_MSE_improves"] = adapted.validation_MSE_ratio_prior.mean() < control_runs.validation_MSE_ratio_prior.mean()
        decision = dict(checks=checks, research_followup_allowed=all(checks.values()), details=details,
            formal_promotion=False, ppo_training_started=False, test_labels_used=False,
            next_step="freeze separate research followup" if all(checks.values()) else "stop; no automatic search or threshold relaxation")
        save(out / "decision.json", decision)
        lines = ["# V60 编码器监督适配结果", "", "12组配对T/V实验完成；未使用G/test选参数，无正式晋升或PPO训练。", "",
            "| 模式 | 平均MSE相对强先验改善 |", "|---|---|"]
        for mode in cfg["modes"]:
            f = table[table['mode'].eq(mode)]
            lines.append(f"| {mode} | {f.improvement_vs_prior.mean():.2%} |")
        lines += ["", f"下一轮研究资格：{decision['research_followup_allowed']}。", "未通过项："+"、".join(k for k,v in checks.items() if not v),
            "", "这是历史验证而非当期可交易views；不得用验证预测回填回测。MSE/RankIC不等于组合夏普。各组changed_parameters和frozen_boundary_passed见summary.json，前缀重放核验在prefix.pt。",
            "", "原V58/V59保留，执行现金流、REIT待查记录、代理和成本认证阻断保持。"]
        (out / "experiment_review.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
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
