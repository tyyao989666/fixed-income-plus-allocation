"""Apply original prediction thresholds to the preregistered multitask candidate."""
import json
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from scripts.run_v52_execution_preflight import save
from src.v57_multitask import require_execution_certificate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",choices=["temporal_opportunity_v55","corporate_action_v58"],default="temporal_opportunity_v55")
    parser.add_argument("--branch",choices=["multitask_v57","multitask_v58"],default="multitask_v57")
    args=parser.parse_args()
    out=ROOT/"output/reports"/args.branch
    cfg=engine.configuration(ROOT/"config/v45_research.yaml")
    with np.load(ROOT/"data_processed"/args.parent/"dataset.npz") as z:
        a={k:z[k] for k in z.files}
    history=[];gates=[]
    for year in range(2022,2027):
        split=engine.shared_split(a,year,cfg)
        bundles=[];summaries=[]
        for seed in cfg["seeds"]:
            root=ROOT/f"output/models/{args.branch}/{year}/{seed}"
            summaries.append(json.loads((root/"summary.json").read_text()))
            with np.load(root/"forecast_arrays.npz") as z: bundles.append({k:z[k] for k in z.files})
        base_model=torch.load(ROOT/f"output/models/{args.parent}/{year}/seed_17_F1.pt",map_location="cpu",weights_only=False)["baseline"]
        bases={part:base_model.predict(engine.raw_features(a,split[part])) for part in ["validation","gate","test"]}
        gy,gm=engine.targets_at(a,split["gate"])
        selected=[{task:b["gate_"+task] for task in ["asset","sleeve"]} for b in bundles]
        component=engine.component_series(engine.ensemble(selected),bases["gate"],gy,gm,a["module_index"])
        parent=json.loads((ROOT/f"output/models/{args.parent}/{year}/report.json").read_text())
        counts=parent["residual_training_dates"]
        available=all(s["validation_relative_to_F0_score"] is not None for s in summaries)
        cutoff=engine.dates_ns(a["dates"])[split["cutoff_index"]]
        for key in component:
            past=[r for r in history if tuple(r["key"])==key and r["year"]<year and 0<r["label_end"]<=cutoff]
            if available:
                g=engine.component_gate(key,selected,bases["gate"],gy,gm,a["module_index"],past,cfg,
                    bool(split["valid"]),counts.get(str(key),0)>=cfg["minimum_train_days"])
            else:
                g=dict(active=False,reason="validation_unavailable_no_gate_authorization",checks={})
            gates.append(dict(year=year,key=list(key),**g))
        ty,tm=engine.targets_at(a,split["test"])
        pred=engine.ensemble([{t:b["test_"+t] for t in ["asset","sleeve"]} for b in bundles])
        series=engine.component_series(pred,bases["test"],ty,tm,a["module_index"])
        ends=engine.end_times(a)
        for key,(ic,delta) in series.items():
            for pos,i in enumerate(split["test"]):
                history.append(dict(key=list(key),year=year,label_end=int(ends[i,key[2]]),ic=float(ic[pos]),delta=float(delta[pos])))
    engine.write_json(out/"forecast_gates.json",dict(gates=gates,original_thresholds_preserved=True,
        inherited_residual_history_requirement_conservative=True,
        active=sum(bool(g["active"]) for g in gates),total=len(gates),
        units="V55 risk-normalized opportunity; raw-return promotion remains separate",
        formal_portfolio_promotion=False))
    ledger=json.loads((out/"ledger_readiness.json").read_text())
    certificate_path=out/"execution_certificate.json"
    certificate_error=None
    certified=False
    try:
        certificate=json.loads(certificate_path.read_text()) if certificate_path.exists() else None
        meta=json.loads((ROOT/"output/reports"/args.parent/"metadata.json").read_text())
        require_execution_certificate(certificate,ledger["dataset_sha256"],meta["assets"],ROOT)
        if pd.Timestamp(certificate["coverage_start"])>pd.Timestamp(meta["start"]) or pd.Timestamp(certificate["coverage_end"])<pd.Timestamp(meta["end"]):
            raise ValueError("Execution certificate period does not cover dataset")
        certified=True
    except (ValueError,KeyError,TypeError) as exc:
        certificate_error=str(exc)
    save(out/"execution_readiness.json",dict(forecast_components_passed=sum(bool(g["active"]) for g in gates),
        ledger_certified=certified,certificate_error=certificate_error,
        formal_ppo_allowed=False,ppo_training_started=False,
        reason="38_asset_execution_certificate_and_new_EP_targets_require_review; component gate alone is insufficient",
        actions="daily decisions: hold / execute25% / execute50% / execute100%; no fixed holding horizon"))
    status=json.loads((out/"pipeline_status.json").read_text())
    status.update(status="research_complete_execution_blocked",forecast_training_started=True,
        completed_forecast_tasks=15,forecast_gate_assessed=True,
        active_forecast_components=sum(bool(g["active"]) for g in gates),
        ppo_training_started=False,formal_promotion=False)
    save(out/"pipeline_status.json",status)
    try:
        from scripts.audit_v57_training import main as audit_artifacts
        audit_artifacts(args.branch,args.parent)
        status["artifact_audit_passed"]=True
        save(out/"pipeline_status.json",status)
    except Exception as exc:
        status.update(status="failed_artifact_audit",artifact_audit_passed=False,error=str(exc))
        save(out/"pipeline_status.json",status)
        raise
    print("V57 prediction gates assessed; formal PPO remains evidence-blocked",flush=True)


if __name__=="__main__":main()
