"""Isolated CUDA objective challenger, with measured CPU/CUDA throughput."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation,source_hash
from src.v53_unified_data import validate_relative_dataset
from src.v55_learning_objective import decode_opportunity
from scripts.run_v52_execution_preflight import save,sha
from scripts.run_v45_research_pipeline import process_alive


@contextmanager
def honest_score_report_units():
    original=engine.write_json
    def writer(path,value):
        if Path(path).name in ["v45_forecast_report.json","v45_forecast_gate.json"]:
            value=dict(value,unit="risk_normalized_past_debiased_opportunity_NOT_return",
                portfolio_stage_allowed=False,
                formal_status="research_objective_only_requires_decoded_views_and_execution_audit")
        return original(path,value)
    engine.write_json=writer
    try:yield
    finally:engine.write_json=original


def benchmark(a,cfg):
    results=[]
    torch.set_num_threads(cfg["threads"])
    for device_name in ["cpu","cuda"]:
        torch.manual_seed(17)
        device=torch.device(device_name)
        with training_implementation():
            model=engine.ForecastNetwork(a,cfg).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["pretrain_lr"])
            indices=a["sample_indices"][-cfg["pretrain_batch"]:]
            times=[]
            for step in range(8):
                if device.type=="cuda":torch.cuda.synchronize()
                start=time.perf_counter()
                batch=engine.batch_at(a,indices,cfg,device)
                optimizer.zero_grad(set_to_none=True)
                loss=model.reconstruction_loss(batch)
                if not torch.isfinite(loss):raise FloatingPointError("Benchmark nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if device.type=="cuda":torch.cuda.synchronize()
                if step>=3:times.append(time.perf_counter()-start)
            results.append(dict(device=device_name,batch_size=len(indices),steps=len(times),
                median_seconds_per_batch=float(np.median(times)),
                includes="window extraction,transfer,forward,backward,optimizer; excludes validation/checkpoint/OOF",
                peak_GPU_allocated_MB=torch.cuda.max_memory_allocated()/2**20 if device_name=="cuda" else None))
        del model,optimizer
        if device_name=="cuda":torch.cuda.empty_cache()
    return dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        batch_results=results,measured_batch_speedup=results[0]["median_seconds_per_batch"]/results[1]["median_seconds_per_batch"],
        total_training_speedup_not_assumed=True,CPU_comparator_running_concurrently=True)


def decode_predictions(a,meta,input_path,output_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    indices={pd.Timestamp(d):i for i,d in enumerate(pd.to_datetime(a["dates"],unit="ns"))}
    names={"asset":{name:i for i,name in enumerate(meta["assets"])},
           "sleeve":{name:i for i,name in enumerate(meta["modules"])}}
    writer=None
    try:
        for batch in pq.ParquetFile(input_path).iter_batches(batch_size=50000):
            frame=batch.to_pandas()
            frame["prediction_units"]="raw_unannualized_relative_return"
            for task in ["asset","sleeve"]:
                for horizon,j in [(5,0),(20,1),(60,2)]:
                    take=frame.task.eq(task)&frame.horizon_days.eq(horizon)
                    part=frame.loc[take]
                    t=np.array([indices[pd.Timestamp(d)] for d in part.as_of_date])
                    e=np.array([names[task][name] for name in part.entity])
                    scale=a[task+"_opportunity_scale"][t,j,e]
                    offset=a[task+"_opportunity_offset"][t,j,e]
                    for column in ["prediction","baseline_prediction","gated_prediction"]:
                        frame.loc[take,column]=decode_opportunity(part[column].to_numpy(),scale,offset)
                    for column in ["active_increment","raw_increment"]:
                        frame.loc[take,column]=np.where(scale>0,part[column].to_numpy()*scale,np.nan)
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(output_path,table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--benchmark",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; use the isolated .venv-gpu-v55 interpreter")
    report=ROOT/"output/reports/temporal_opportunity_v55"
    out=ROOT/"data_processed/temporal_opportunity_v55"
    state=json.loads((report/"status.json").read_text(encoding="utf-8"))
    if sha(out/"dataset.npz")!=state["dataset_sha256"] or sha(report/"metadata.json")!=state["metadata_sha256"]:
        raise ValueError("Objective dataset registration mismatch")
    meta=json.loads((report/"metadata.json").read_text(encoding="utf-8"))
    for item in meta["source_manifest"].values():
        if sha(ROOT/item["path"])!=item["sha256"]:raise ValueError("Source mismatch: "+item["path"])
    with np.load(out/"dataset.npz",allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    validate_relative_dataset(a)
    cfg=engine.configuration(ROOT/"config/v45_research.yaml")
    cfg["seed_workers"]=1
    cfg["v55_implementation"]=dict(objective_sha256=sha(ROOT/"src/v55_learning_objective.py"),
        dataset_sha256=state["dataset_sha256"],mask_safe_hash=source_hash(),device="cuda",score_units="temporal_opportunity")
    if args.benchmark:
        result=benchmark(a,cfg);save(report/"GPU_benchmark.json",result);print(json.dumps(result,indent=2),flush=True)
        if not args.train:return 0
    lock=report/"active_training.json"
    if lock.exists() and process_alive(json.loads(lock.read_text(encoding="utf-8"))["pid"]):
        raise RuntimeError("A V55 CUDA fit is already active")
    progress=dict(status="preflight",pid=os.getpid(),device="cuda",gpu=torch.cuda.get_device_name(0),
        dataset_sha256=state["dataset_sha256"],training_started=False,formal_promotion=False,ppo_started=False,
        years=cfg["years"],seeds=cfg["seeds"],max_epochs=cfg["max_epochs"],pretrain_max_epochs=cfg["pretrain_max_epochs"])
    save(report/"training_status.json",progress)
    preflight=engine.forecast_preflight(a,cfg);engine.write_preflight(report,preflight)
    unavailable=[f["year"] for f in preflight["folds"] if not f["structural_selection_available"]]
    if unavailable:
        progress.update(status="blocked_structural_validation",unavailable_years=unavailable)
        save(report/"training_status.json",progress);return 2
    if not args.train:
        progress["status"]="preflight_passed";save(report/"training_status.json",progress);return 0
    save(lock,dict(pid=os.getpid()))
    try:
        progress.update(status="training",training_started=True);save(report/"training_status.json",progress)
        with training_implementation(),honest_score_report_units():
            engine.run_training(a,meta,cfg,ROOT/"output/models/temporal_opportunity_v55",out/"score_predictions.parquet",report,
                resume=args.resume,debug=False,device="cuda")
        decode_predictions(a,meta,out/"score_predictions.parquet",out/"decoded_relative_return_views.parquet")
        progress.update(status="completed_objective_research",decoded_views_available=True,formal_promotion=False)
        save(report/"training_status.json",progress)
    except BaseException as exc:
        progress.update(status="failed",error=f"{type(exc).__name__}: {exc}");save(report/"training_status.json",progress);raise
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding="utf-8"))["pid"]==os.getpid():lock.unlink()
    return 0


if __name__=="__main__":sys.exit(main())
