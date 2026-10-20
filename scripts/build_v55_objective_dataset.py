"""Create a new objective branch; never overwrites active CPU comparator inputs."""
from pathlib import Path
import copy
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v55_learning_objective import temporal_opportunity_targets, decode_opportunity
from src.v53_unified_data import validate_relative_dataset
from src.v45_data import annual_splits
from scripts.run_v52_execution_preflight import sha,save


def main():
    out=ROOT/"data_processed/temporal_opportunity_v55"
    report=ROOT/"output/reports/temporal_opportunity_v55"
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    parent=ROOT/"data_processed/unified_v53/dataset.npz"
    parent_meta=ROOT/"output/reports/unified_v53/metadata.json"
    registration=json.loads((ROOT/"output/reports/unified_v53/status.json").read_text(encoding="utf-8"))
    if sha(parent)!=registration["dataset_sha256"] or sha(parent_meta)!=registration["metadata_sha256"]:
        raise ValueError("Parent data hash mismatch")
    with np.load(parent,allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    metadata=json.loads(parent_meta.read_text(encoding="utf-8"))
    observed=np.where(a["return_masks"],a["returns"],np.nan)
    sw=a["sleeve_reference_weights"]
    sleeve=np.sum(sw* np.where(a["return_masks"],a["returns"],0)[:,None,:],axis=-1)
    sleeve_ok=(sw.sum(-1)>0)&((sw==0)|a["return_masks"][:,None,:]).all(-1)
    ref_path=ROOT/"data_processed/unified_v53/cb_reference_returns.parquet"
    sleeve[:,2]=pd.read_parquet(ref_path).iloc[:,0].to_numpy()
    sleeve_ok[:,2]=np.isfinite(sleeve[:,2])
    sleeve=np.where(sleeve_ok,sleeve,np.nan)
    audits=[]
    for task,daily in [("asset",observed),("sleeve",sleeve)]:
        result=temporal_opportunity_targets(a[task+"_target"],a[task+"_target_mask"],daily)
        a[task+"_raw_relative_target"]=a[task+"_target"].copy()
        a[task+"_raw_relative_target_mask"]=a[task+"_target_mask"].copy()
        a[task+"_target"],a[task+"_target_mask"]=result["target"],result["mask"]
        a[task+"_opportunity_scale"],a[task+"_opportunity_offset"]=result["scale"],result["offset"]
        restored=decode_opportunity(result["target"],result["scale"],result["offset"])
        good=result["mask"]
        if not np.allclose(restored[good],a[task+"_raw_relative_target"][good],rtol=1e-5,atol=1e-6):
            raise ValueError("Objective cannot be decoded to raw relative-return views")
        for j,h in enumerate([5,20,60]):
            audits.append(dict(task=task,horizon=h,valid_cells=int(good[:,j].sum()),
                original_cells=int(a[task+"_raw_relative_target_mask"][:,j].sum()),
                invalid_tail=int(good[-h:,j].sum())))
    validate_relative_dataset(a)
    np.savez_compressed(out/"dataset.npz",**a)
    metadata=copy.deepcopy(metadata)
    metadata.update(version="v55_causal_temporal_opportunity",prediction_units="risk_normalized_past_debiased_opportunity",
        objective=dict(vol_window=60,vol_min_observations=40,annual_vol_floor=.01,
            location_window=252,location_min_matured_labels=126,location_shrinkage_count=252,
            location_latest_label_start="t-h-1",horizons=[5,20,60],
            formula="raw_relative_return/(past_daily_vol*sqrt(h))-shrunk_past_matured_score_mean",
            inverse="raw_relative_return=(score+known_offset)*known_scale",
            note="Scores are NOT return views. EP/PPO must use decoded raw relative-return fields."),
        parent_dataset_sha256=sha(parent),formal_promotion=False,
        source_manifest={**metadata["source_manifest"],"v55_parent":dict(path=str(parent.relative_to(ROOT)),sha256=sha(parent)),
            "v55_objective":dict(path="src/v55_learning_objective.py",sha256=sha(ROOT/"src/v55_learning_objective.py")),
            "v55_builder":dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(Path(__file__))),
            "v55_cb_reference":dict(path=str(ref_path.relative_to(ROOT)),sha256=sha(ref_path))})
    metadata["label_policy"]=dict(version="v55_temporal_opportunity",parent=metadata["label_policy"],
        units="dimensionless_opportunity",gaps="preserved; no label or risk backfill")
    metadata["array_schema"]={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in a.items()}
    save(report/"metadata.json",metadata)
    pd.DataFrame(audits).to_csv(report/"objective_coverage.csv",index=False)
    splits=[]
    for year in range(2022,2027):
        split=annual_splits(a,year)
        splits.append(dict(year=year,reason=split["reason"],**{k:len(split[k]) for k in ["train","validation","gate","test"]}))
    pd.DataFrame(splits).to_csv(report/"annual_splits.csv",index=False)
    status=dict(status="objective_dataset_built",dataset_sha256=sha(out/"dataset.npz"),metadata_sha256=sha(report/"metadata.json"),
        parent_dataset_sha256=sha(parent),assets=38,nine_grid_preserved=True,max_epochs=1000,
        models_trained=False,formal_promotion=False,splits=splits)
    save(report/"status.json",status);print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
