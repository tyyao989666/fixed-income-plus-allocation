"""Regression checks for the actual two return-basis incidents; no P&L claim."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha,save


def main():
    report=ROOT/"output/reports/corporate_action_v58"
    state=json.loads((report/"status.json").read_text())
    meta=json.loads((report/"metadata.json").read_text())
    with np.load(ROOT/"data_processed/corporate_action_v58/dataset.npz") as z:a={k:z[k] for k in z.files}
    with np.load(ROOT/"data_processed/unified_v53/dataset.npz") as z:old={k:z[k] for k in ["returns","return_masks","dates","macro_values","macro_masks"]}
    dates=pd.DatetimeIndex(pd.to_datetime(a["dates"],unit="ns"));assets=meta["assets"]
    split=dates.get_loc("2023-01-16");gap=dates.get_loc("2026-05-15")
    aaa=assets.index("credit_lgfv_aaa");aa=assets.index("credit_lgfv_aa")
    checks={
        "dataset_hash":sha(ROOT/"data_processed/corporate_action_v58/dataset.npz")==state["dataset_sha256"],
        "assets38":len(assets)==38,
        "same_calendar":np.array_equal(a["dates"],old["dates"]),
        "split_adjusted_not_clipped":np.isclose(a["returns"][split,aaa],9.469*10/95.318-1,atol=1e-7),
        "unknown_basis_return_masked":not a["return_masks"][gap,aa] and a["returns"][gap,aa]==0,
        "only_one_return_mask_removed":int((old["return_masks"]&~a["return_masks"]).sum())==1,
        "no_new_observations_fabricated":not (a["return_masks"]&~old["return_masks"]).any(),
        "nine_grid_preserved":np.array_equal(a["macro_values"],old["macro_values"]) and np.array_equal(a["macro_masks"],old["macro_masks"]),
        "no_unadjudicated_gross_moves":not (np.abs(a["returns"][a["return_masks"]])>.25).any()}
    good=old["return_masks"]&a["return_masks"]
    differences=good & ~np.isclose(a["returns"],old["returns"],rtol=0,atol=1e-7)
    checks["only_split_changes_observed_return"]=int(differences.sum())==1 and bool(differences[split,aaa])
    for j,h in enumerate([5,20,60]):
        checks[f"no_valid_{h}d_labels_cross_unknown_gap"]=not a["asset_raw_relative_target_mask"][gap-h:gap,j,aa].any()
        checks[f"unmatured_{h}d_tail_masked"]=not a["asset_target_mask"][-h:,j].any() and not a["sleeve_target_mask"][-h:,j].any()
    for item in meta["source_manifest"].values():
        if sha(ROOT/item["path"])!=item["sha256"]:raise ValueError("Source changed")
    checks={k:bool(v) for k,v in checks.items()}
    save(report/"data_audit.json",dict(passed=bool(all(checks.values())),checks=checks,
        scope="two basis incidents, masks, chronology, grid and hashes; NOT full cashflow certification",
        formal_promotion=False))
    if not all(checks.values()):raise ValueError([k for k,v in checks.items() if not v])
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
