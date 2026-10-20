"""Fresh corrected-data encoder training -> distribution heads -> audited PPO block."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation,source_hash
from src.v56_total_view_ranking import total_view_ranking_implementation
from src.windows_checkpoint_io import checkpoint_io
from scripts.run_v55_gpu_objective import decode_predictions,honest_score_report_units
from scripts.run_v52_execution_preflight import save,sha
from scripts.run_v45_research_pipeline import process_alive


def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA interpreter required")
    parent="corporate_action_v58";branch="multitask_v58"
    report=ROOT/"output/reports"/parent
    data=ROOT/"data_processed"/parent
    state=json.loads((report/"status.json").read_text())
    audit=json.loads((report/"data_audit.json").read_text())
    if not audit["passed"]:raise ValueError("Corrected-data audit has not passed")
    assert sha(data/"dataset.npz")==state["dataset_sha256"]
    meta=json.loads((report/"metadata.json").read_text())
    for item in meta["source_manifest"].values():assert sha(ROOT/item["path"])==item["sha256"]
    with np.load(data/"dataset.npz",allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if (np.abs(a["returns"][a["return_masks"]])>.25).any():raise ValueError("Unadjudicated extreme return")
    cfg=engine.configuration(ROOT/"config/v45_research.yaml")
    cfg["seed_workers"]=1
    cfg["v58_implementation"]=dict(dataset_sha256=state["dataset_sha256"],mask_safe_sha256=source_hash(),
        ranking_sha256=sha(ROOT/"src/v56_total_view_ranking.py"),runner_sha256=sha(Path(__file__)),
        design_sha256=sha(ROOT/"docs/v58_data_integrity_rebuild.md"),
        checkpoint_io_sha256=sha(ROOT/"src/windows_checkpoint_io.py"),
        scope="fresh_SSL_and_supervised_models; never use retired historical encoders")
    lock=report/"active_training.json"
    if lock.exists() and process_alive(json.loads(lock.read_text())["pid"]):raise RuntimeError("V58 already active")
    progress=dict(status="training_fresh_forecast",pid=os.getpid(),stage="fresh_iTransformer",formal_promotion=False,
                  ppo_training_started=False,provenance=cfg["v58_implementation"])
    save(lock,dict(pid=os.getpid()));save(report/"pipeline_status.json",progress)
    try:
        with training_implementation(),total_view_ranking_implementation(engine),honest_score_report_units(),checkpoint_io(engine):
            engine.run_training(a,meta,cfg,ROOT/"output/models"/parent,data/"score_predictions.parquet",report,
                                resume=True,debug=False,device="cuda")
        decode_predictions(a,meta,data/"score_predictions.parquet",data/"decoded_relative_return_views.parquet")
        progress.update(status="training_distribution_heads",stage="multitask_heads")
        save(report/"pipeline_status.json",progress)
        target_report=ROOT/"output/reports"/branch;target_report.mkdir(parents=True,exist_ok=True)
        ledger=json.loads((ROOT/"output/reports/multitask_v57/ledger_readiness.json").read_text())
        ledger.update(dataset_sha256=state["dataset_sha256"],corporate_action_incidents_corrected_or_masked=True)
        save(target_report/"ledger_readiness.json",ledger)
        save(target_report/"pipeline_status.json",dict(status="fresh_parent_complete_training_heads",formal_promotion=False,ledger=ledger))
        for script in ["train_v57_multitask.py","gate_v57_forecasts.py"]:
            subprocess.run([sys.executable,"-u","-X","utf8",str(ROOT/"scripts"/script),"--parent",parent,"--branch",branch],cwd=ROOT,check=True)
        progress.update(status="research_complete_execution_blocked",stage="finished",formal_promotion=False)
        save(report/"pipeline_status.json",progress)
    except BaseException as exc:
        progress.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        save(report/"pipeline_status.json",progress);raise
    finally:
        if lock.exists() and json.loads(lock.read_text())["pid"]==os.getpid():lock.unlink()


if __name__=="__main__":main()
