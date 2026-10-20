"""Descriptive T/V distribution shift; never changes scaling or model selection."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src import v45_forecast as engine


def main():
    meta=json.loads((ROOT/"output/reports/temporal_opportunity_v55/metadata.json").read_text())
    cfg=engine.configuration(ROOT/"config/v45_research.yaml")
    with np.load(ROOT/"data_processed/temporal_opportunity_v55/dataset.npz") as z:a={k:z[k] for k in z.files}
    rows=[];features=[]
    for year in range(2022,2027):
        split=engine.shared_split(a,year,cfg)
        t,v=split["train"],split["validation"]
        for task,names in [("asset",meta["assets"]),("sleeve",meta["modules"])]:
            for j,h in enumerate([5,20,60]):
                for e,name in enumerate(names):
                    for kind,key,maskkey in [("opportunity",task+"_target",task+"_target_mask"),
                                             ("raw_relative",task+"_raw_relative_target",task+"_raw_relative_target_mask")]:
                        train=a[key][t,j,e][a[maskkey][t,j,e]]
                        valid=a[key][v,j,e][a[maskkey][v,j,e]]
                        if len(train)<20 or len(valid)<10:continue
                        qlo,qhi=np.quantile(train,[.01,.99])
                        rows.append(dict(year=year,task=task,entity=name,horizon=h,kind=kind,
                            train_days=len(train),validation_days=len(valid),
                            train_std=float(train.std()),validation_std=float(valid.std()),
                            validation_to_train_std=float(valid.std()/max(train.std(),1e-8)),
                            validation_outside_train_1_99=float(((valid<qlo)|(valid>qhi)).mean()),
                            validation_constant_train_mean_MSE=float(np.mean((valid-train.mean())**2)),
                            maximum_absolute_validation_label=float(np.max(np.abs(valid)))))
        for e,name in enumerate(meta["assets"]):
            for f,feature in enumerate(meta["local_features"]):
                x=a["local_values"][t,e,f][a["local_masks"][t,e,f]]
                y=a["local_values"][v,e,f][a["local_masks"][v,e,f]]
                if len(x)<20 or len(y)<10:continue
                lo,hi=np.quantile(x,[.01,.99])
                features.append(dict(year=year,entity=name,feature=feature,train_days=len(x),validation_days=len(y),
                    validation_outside_train_1_99=float(((y<lo)|(y>hi)).mean())))
    out=ROOT/"output/reports/multitask_v57"
    pd.DataFrame(rows).to_csv(out/"label_distribution_shift.csv",index=False)
    pd.DataFrame(features).to_csv(out/"feature_distribution_shift.csv",index=False)
    d=pd.DataFrame(rows)
    print(d[d.year.eq(2024)&d.kind.eq("opportunity")].nlargest(5,"validation_constant_train_mean_MSE")
          [["entity","horizon","validation_to_train_std","validation_constant_train_mean_MSE"]].to_string(index=False))


if __name__=="__main__":main()
