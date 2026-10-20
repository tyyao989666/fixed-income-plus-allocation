"""Frozen historical iTransformer transfer + nonlinear distribution/risk heads.

No portfolio training: uncertified reward histories remain blocked.
"""
import copy
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation
from src.v57_multitask import DistributionHead, multitask_loss, future_downside
from scripts.run_v52_execution_preflight import sha, save
from scripts.run_v45_research_pipeline import process_alive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",choices=["temporal_opportunity_v55","corporate_action_v58"],default="temporal_opportunity_v55")
    parser.add_argument("--branch",choices=["multitask_v57","multitask_v58"],default="multitask_v57")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu-v55 CUDA interpreter")
    out = ROOT/"output/reports"/args.branch
    models = ROOT/"output/models"/args.branch
    data = ROOT/"data_processed"/args.branch
    for p in [out,models,data]: p.mkdir(parents=True,exist_ok=True)
    lock = out/"active_training.json"
    if lock.exists() and process_alive(json.loads(lock.read_text())["pid"]):
        raise RuntimeError("V57 already active")
    cfg = json.loads((ROOT/"config/v57_multitask.json").read_text())
    oldcfg = engine.configuration(ROOT/"config/v45_research.yaml")
    oldcfg["seed_workers"] = 1
    torch.set_num_threads(2)
    device = torch.device("cuda")
    parent = ROOT/"output/reports"/args.parent
    meta = json.loads((parent/"metadata.json").read_text())
    registration = json.loads((parent/"status.json").read_text())
    dataset = ROOT/"data_processed"/args.parent/"dataset.npz"
    assert sha(dataset) == registration["dataset_sha256"]
    registry=json.loads((ROOT/"output/reports/data_integrity_registry.json").read_text())
    if sha(dataset) in registry["retired_dataset_hashes"]:
        raise ValueError("Source dataset retired for corporate-action error; old encoders cannot be transferred")
    for source in meta["source_manifest"].values():
        assert sha(ROOT/source["path"]) == source["sha256"]
    with np.load(dataset,allow_pickle=False) as z: a = {k:z[k].copy() for k in z.files}
    assert len(meta["assets"]) == 38 and meta["nine_grid_preserved"]
    own = np.where(a["return_masks"],a["returns"],np.nan)
    weights = a["sleeve_reference_weights"]
    sr = np.sum(weights*np.where(a["return_masks"],a["returns"],0)[:,None,:],axis=-1)
    sm = (weights.sum(-1)>0) & ((weights==0)|a["return_masks"][:,None,:]).all(-1)
    sr[:,2] = pd.read_parquet(ROOT/"data_processed/unified_v53/cb_reference_returns.parquet").iloc[:,0].to_numpy()
    sm[:,2] = np.isfinite(sr[:,2])
    risk, risk_mask = {}, {}
    for task,r,m in [("asset",own,a["return_masks"]),("sleeve",sr,sm)]:
        dd, good = future_downside(r,m)
        scale = a[task+"_opportunity_scale"]
        risk_mask[task] = good & (scale>0) & a[task+"_target_mask"]
        risk[task] = np.divide(dd,scale,out=np.zeros_like(dd),where=scale>0)
    provenance = dict(dataset_sha256=sha(dataset),config_sha256=sha(ROOT/"config/v57_multitask.json"),
        module_sha256=sha(ROOT/"src/v57_multitask.py"),runner_sha256=sha(Path(__file__)),
        preregistration_sha256=sha(ROOT/("docs/v58_data_integrity_rebuild.md" if args.parent=="corporate_action_v58" else "docs/v57_rebuild_design.md")))
    status = dict(status="training",pid=os.getpid(),completed_tasks=0,total_tasks=15,
                  formal_promotion=False,ppo_training_started=False,provenance=provenance,
                  phase=cfg["phase"],pretrained_encoder_frozen=True)
    save(lock,dict(pid=os.getpid()))
    save(out/"training_status.json",status)
    summaries = []
    try:
        with training_implementation():
            for year in cfg["years"]:
                split = engine.shared_split(a,year,oldcfg)
                if not split["valid"]: raise ValueError("Invalid original split")
                clipped,_ = engine.clip_fold_features(a,split["train"],oldcfg)
                for seed in cfg["seeds"]:
                    fold = models/str(year)/str(seed)
                    fold.mkdir(parents=True,exist_ok=True)
                    source = ROOT/f"output/models/{args.parent}/{year}/seed_{seed}_pretraining.pt"
                    identity = dict(**provenance,encoder_checkpoint_sha256=sha(source),year=year,seed=seed)
                    done = fold/"summary.json"
                    if done.exists():
                        previous = json.loads(done.read_text())
                        if previous["identity"] != identity: raise ValueError("Resume identity mismatch")
                        summaries.append(previous)
                        status["completed_tasks"] = len(summaries)
                        save(out/"training_status.json",status)
                        continue
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    encoder = engine.ForecastNetwork(clipped,oldcfg).to(device)
                    saved = torch.load(source,map_location=device,weights_only=False)
                    encoder.load_state_dict(saved["best_state"])
                    if not saved.get("complete"): raise ValueError("Parent encoder incomplete")
                    encoder.eval()
                    features = {name:engine.infer(encoder,clipped,split[name],oldcfg,device,True)
                                for name in ["train","validation","gate","test"]}
                    del encoder,saved
                    train_features = {k:torch.tensor(v,device=device) for k,v in features["train"].items()}
                    val_features = {k:torch.tensor(v,device=device) for k,v in features["validation"].items()}
                    baseline = torch.load(ROOT/f"output/models/{args.parent}/{year}/seed_{seed}_F1.pt",map_location="cpu",weights_only=False)["baseline"]
                    validation_base = baseline.predict(engine.raw_features(a,split["validation"]))
                    modules = {"asset":torch.tensor(a["module_index"],device=device),
                               "sleeve":torch.zeros(5,dtype=torch.long,device=device)}
                    heads = nn.ModuleDict({t:DistributionHead(oldcfg["d_model"]) for t in ["asset","sleeve"]}).to(device)
                    opt = torch.optim.AdamW(heads.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
                    tensors = {}
                    for name in ["train","validation"]:
                        ix = split[name]
                        tensors[name] = {t:tuple(torch.tensor(v,device=device) for v in
                            [a[t+"_target"][ix],a[t+"_target_mask"][ix],risk[t][ix],risk_mask[t][ix]])
                            for t in heads}
                    rng = np.random.default_rng(seed)
                    best,best_epoch = float("inf"),0
                    history = []
                    best_state = copy.deepcopy(heads.state_dict())
                    for epoch in range(1,cfg["maximum_epochs"]+1):
                        heads.train();losses=[]
                        order = rng.permutation(len(split["train"]))
                        for start in range(0,len(order),cfg["batch_size"]):
                            ix = order[start:start+cfg["batch_size"]]
                            opt.zero_grad(set_to_none=True)
                            loss = torch.zeros((),device=device)
                            for task in heads:
                                q,d,mean = heads[task](train_features[task][ix])
                                values = [v[ix] for v in tensors["train"][task]]
                                loss = loss+.5*multitask_loss(q,d,mean,*values,modules[task])
                            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite multitask loss")
                            loss.backward();nn.utils.clip_grad_norm_(heads.parameters(),1.,error_if_nonfinite=True)
                            opt.step();losses.append(float(loss.detach()))
                        if epoch==1 or epoch%cfg["validation_every"]==0:
                            heads.eval()
                            with torch.no_grad():
                                val = sum(.5*multitask_loss(*heads[t](val_features[t]),*tensors["validation"][t],modules[t]) for t in heads)
                            value = float(val)
                            if not np.isfinite(value): raise FloatingPointError("Nonfinite validation loss")
                            if value < best-1e-4:
                                best,best_epoch,best_state = value,epoch,copy.deepcopy(heads.state_dict())
                            row = dict(year=year,seed=seed,epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=value,best_epoch=best_epoch)
                            history.append(row)
                            pd.DataFrame(history).to_csv(fold/"loss.csv",index=False)
                            status.update(year=year,seed=seed,epoch=epoch,latest_loss=row)
                            save(out/"training_status.json",status)
                            print(json.dumps(row),flush=True)
                            if epoch>=cfg["minimum_epochs"] and epoch-best_epoch>=cfg["patience_epochs"]: break
                    heads.load_state_dict(best_state);heads.eval()
                    torch.save(dict(identity=identity,state_dict=best_state,selected_epoch=best_epoch),fold/"model.pt")
                    metrics=[];prediction=[];forecast_arrays={}
                    with torch.no_grad():
                        for part in ["validation","gate","test"]:
                            for task in heads:
                                q,d,mean = heads[task](torch.tensor(features[part][task],device=device))
                                q,d,mean = q.cpu().numpy(),d.cpu().numpy(),mean.cpu().numpy()
                                forecast_arrays[part+"_"+task] = mean
                                forecast_arrays[part+"_"+task+"_quantiles"] = q
                                ix=split[part];scale=a[task+"_opportunity_scale"][ix];offset=a[task+"_opportunity_offset"][ix]
                                valid=a[task+"_target_mask"][ix]
                                if part=="validation":
                                    truth=a[task+"_target"][ix]
                                    metrics.append(dict(task=task,interval_coverage=float(((truth>=q[...,0])&(truth<=q[...,2]))[valid].mean()),
                                        median_score_MAE=float(np.abs(truth-q[...,1])[valid].mean()),
                                        note="nominal80 interval; empirical validation coverage, not calibrated confidence"))
                                if part=="test":
                                    names=meta["assets"] if task=="asset" else meta["modules"]
                                    for j,h in enumerate([5,20,60]):
                                        for e,name in enumerate(names):
                                            known=scale[:,j,e]>0
                                            for k,t in enumerate(ix):
                                                prediction.append(dict(as_of_date=pd.Timestamp(a["dates"][t]),task=task,entity=name,horizon_days=h,seed=seed,
                                                    lower=(q[k,j,e,0]+offset[k,j,e])*scale[k,j,e] if known[k] else np.nan,
                                                    mean=(mean[k,j,e]+offset[k,j,e])*scale[k,j,e] if known[k] else np.nan,
                                                    median=(q[k,j,e,1]+offset[k,j,e])*scale[k,j,e] if known[k] else np.nan,
                                                    upper=(q[k,j,e,2]+offset[k,j,e])*scale[k,j,e] if known[k] else np.nan,
                                                    downside=d[k,j,e]*scale[k,j,e] if known[k] else np.nan,
                                                    units="raw_relative_return_and_own_price_drawdown",
                                                    deployment_allowed=False,train_end=pd.Timestamp(a["dates"][split["train"][-1]])))
                                if part=="validation":
                                    features["validation"][task]=mean
                    vy,vm=engine.targets_at(a,split["validation"])
                    score=engine.selection_score(features["validation"],validation_base,vy,vm,a["module_index"],oldcfg)
                    summary=dict(identity=identity,selected_epoch=best_epoch,trained_epochs=epoch,validation_loss=best,
                        validation_relative_to_F0_score=float(score) if np.isfinite(score) else None,
                        metrics=metrics,formal_prediction_gate="not_assessed",formal_promotion=False)
                    pd.DataFrame(prediction).to_parquet(fold/"predictions.parquet",index=False)
                    np.savez_compressed(fold/"forecast_arrays.npz",**forecast_arrays)
                    save(done,summary);summaries.append(summary)
                    status["completed_tasks"]=len(summaries);save(out/"training_status.json",status)
                    del features,train_features,val_features,tensors,heads,opt
                    torch.cuda.empty_cache()
        pd.DataFrame([dict(year=s["identity"]["year"],seed=s["identity"]["seed"],selected_epoch=s["selected_epoch"],
                          validation_score=s["validation_relative_to_F0_score"],validation_loss=s["validation_loss"]) for s in summaries]).to_csv(out/"training_summary.csv",index=False)
        status.update(status="completed_research_forecast",formal_prediction_gate="not_assessed",
                      execution_blocked=True,ppo_training_started=False)
        save(out/"training_status.json",status)
    except BaseException as exc:
        status.update(status="failed",error=f"{type(exc).__name__}: {exc}");save(out/"training_status.json",status);raise
    finally:
        if lock.exists() and json.loads(lock.read_text())["pid"]==os.getpid():lock.unlink()


if __name__=="__main__": main()
