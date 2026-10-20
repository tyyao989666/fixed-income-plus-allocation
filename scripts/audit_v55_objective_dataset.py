from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha,save
from src.v55_learning_objective import decode_opportunity


def main():
    report=ROOT/"output/reports/temporal_opportunity_v55"
    out=ROOT/"data_processed/temporal_opportunity_v55"
    state=json.loads((report/"status.json").read_text(encoding="utf-8"))
    meta=json.loads((report/"metadata.json").read_text(encoding="utf-8"))
    with np.load(out/"dataset.npz",allow_pickle=False) as f:a={k:f[k] for k in f.files}
    with np.load(ROOT/"data_processed/unified_v53/dataset.npz",allow_pickle=False) as f:p={k:f[k] for k in f.files}
    checks=[]
    def check(name,value):checks.append(dict(check=name,passed=bool(value)))
    check("dataset_hash",sha(out/"dataset.npz")==state["dataset_sha256"])
    check("metadata_hash",sha(report/"metadata.json")==state["metadata_sha256"])
    check("parent_dataset_unchanged",sha(ROOT/"data_processed/unified_v53/dataset.npz")==state["parent_dataset_sha256"])
    for key in ["dates","local_values","local_masks","macro_values","macro_masks","regime_index","returns","return_masks","label_end","module_index","sample_indices"]:
        check("input_unchanged:"+key,np.array_equal(a[key],p[key]))
    for task in ["asset","sleeve"]:
        good=a[task+"_target_mask"].astype(bool)
        check(task+"_no_unmasked_missing_parent_labels",not (good&~p[task+"_target_mask"].astype(bool)).any())
        check(task+"_finite_targets",np.isfinite(a[task+"_target"]).all())
        check(task+"_false_masks_zero_placeholder",(a[task+"_target"][~good]==0).all())
        decoded=decode_opportunity(a[task+"_target"],a[task+"_opportunity_scale"],a[task+"_opportunity_offset"])
        check(task+"_inverse_to_raw_units",np.allclose(decoded[good],p[task+"_target"][good],rtol=1e-5,atol=1e-6))
        for j,h in enumerate([5,20,60]):check(task+f"_immature_tail_{h}",not good[-h:,j].any())
    for key,item in meta["source_manifest"].items():check("source_hash:"+key,sha(ROOT/item["path"])==item["sha256"])
    pd.DataFrame(checks).to_csv(report/"artifact_checks.csv",index=False)
    result=dict(passed=all(c["passed"] for c in checks),checks=len(checks),
        dataset_sha256=state["dataset_sha256"],formal_promotion=False,
        objective="past-vol-scaled relative return minus matured-only temporal location",
        same_inputs_and_nine_grid=True,portfolio_return_or_reward_not_yet_replaced=True)
    save(report/"audit.json",result);print(json.dumps(result,indent=2))
    if not result["passed"]:raise SystemExit("V55 audit failed")


if __name__=="__main__":main()
