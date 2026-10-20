"""Validate actual completed forecast artifacts, not claimed portfolio performance."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save


def main(branch="multitask_v57",parent="temporal_opportunity_v55"):
    out=ROOT/"output/reports"/branch
    state=json.loads((out/"training_status.json").read_text())
    if state["status"]!="completed_research_forecast" or state["completed_tasks"]!=15:
        raise ValueError("Full forecast has not completed")
    manifest=state["provenance"]
    paths=[("src/v57_multitask.py","module_sha256"),("scripts/train_v57_multitask.py","runner_sha256"),
           ("config/v57_multitask.json","config_sha256"),
           ("docs/v58_data_integrity_rebuild.md" if parent=="corporate_action_v58" else "docs/v57_rebuild_design.md","preregistration_sha256"),
           (f"data_processed/{parent}/dataset.npz","dataset_sha256")]
    for path,key in paths:
        if sha(ROOT/path)!=manifest[key]:raise ValueError("Frozen source changed: "+path)
    rows=[]
    for year in range(2022,2027):
        for seed in [17,29,47]:
            fold=ROOT/f"output/models/{branch}/{year}/{seed}"
            summary=json.loads((fold/"summary.json").read_text())
            if any(summary["identity"][k]!=v for k,v in manifest.items()):
                raise ValueError("Completed task source mismatch")
            original=ROOT/f"output/models/{parent}/{year}/seed_{seed}_pretraining.pt"
            if sha(original)!=summary["identity"]["encoder_checkpoint_sha256"]:
                raise ValueError("Historical encoder changed")
            f=pd.read_parquet(fold/"predictions.parquet")
            if f.duplicated(["as_of_date","task","entity","horizon_days","seed"]).any():
                raise ValueError("Duplicate forecast keys")
            if not (pd.to_datetime(f.train_end)<pd.to_datetime(f.as_of_date)).all():
                raise ValueError("Prediction date isolation failed")
            if f.deployment_allowed.any():raise ValueError("Research forecast promoted without authorization")
            good=f[["lower","median","upper","mean","downside"]].notna().all(axis=1)
            p=f.loc[good]
            if not np.isfinite(p[["lower","median","upper","mean","downside"]]).all().all():
                raise ValueError("Nonfinite observed forecast")
            if not ((p.lower<=p["median"])&(p["median"]<=p.upper)&(p.downside>=0)).all():
                raise ValueError("Invalid quantiles or downside")
            if not f.loc[~good,["lower","median","upper","mean","downside"]].isna().all().all():
                raise ValueError("Partial inverse availability is inconsistent")
            loss=pd.read_csv(fold/"loss.csv")
            if not np.isfinite(loss[["train_loss","validation_loss"]]).all().all():
                raise ValueError("Nonfinite training log")
            if summary["trained_epochs"]>1000 or summary["trained_epochs"]<50:
                raise ValueError("Epoch contract violated")
            rows.append(dict(year=year,seed=seed,rows=len(f),available_rows=int(good.sum()),
                selected_epoch=summary["selected_epoch"],trained_epochs=summary["trained_epochs"],
                validation_score=summary["validation_relative_to_F0_score"]))
    pd.DataFrame(rows).to_csv(out/"artifact_checks.csv",index=False)
    save(out/"artifact_audit.json",dict(passed=True,tasks=len(rows),prediction_rows=sum(r["rows"] for r in rows),
        mean_separate_from_median=True,quantiles_ordered=True,downside_nonnegative=True,
        frozen_encoder_transfer_not_full_encoder_retraining=True,
        no_execution_reward_certification_implied=True,formal_promotion=False))
    print("V57 artifact audit passed",flush=True)


if __name__=="__main__":main()
