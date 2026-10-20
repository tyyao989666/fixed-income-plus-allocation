"""Rebuild all features/labels after two verified historical return-basis incidents."""
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import requests
from src.v45_data import local_features, observable_grid
from src.v53_unified_data import independent_cb_relative_labels, validate_relative_dataset
from src.v55_learning_objective import temporal_opportunity_targets, decode_opportunity
from scripts.run_v52_execution_preflight import save,sha


def main():
    source=ROOT/"data_processed/unified_v53"
    out=ROOT/"data_processed/corporate_action_v58"
    report=ROOT/"output/reports/corporate_action_v58"
    raw=ROOT/"data_raw/corporate_action_v58"
    for p in [out,report,raw]:p.mkdir(parents=True,exist_ok=True)
    url="https://www.sse.com.cn/disclosure/fund/announcement/c/new/2023-01-16/511220_20230116_F9OY.pdf"
    evidence=raw/"reviewed_split_evidence.json"
    if evidence.exists():
        event=json.loads(evidence.read_text())
        if not (event["security"]=="511220.SH" and event["new_units_per_old_unit"]==10
                and event["registration_date"]=="2023-01-13" and event["verified_in_primary_source"]):
            raise ValueError("Reviewed corporate action evidence mismatch")
    else:
        evidence=raw/"511220_split_result_20230116.pdf"
        response=requests.get(url,timeout=30);response.raise_for_status()
        if not response.content.startswith(b"%PDF"):raise ValueError("Split evidence is not PDF")
        evidence.write_bytes(response.content)
        from pypdf import PdfReader
        text="".join(page.extract_text() or "" for page in PdfReader(evidence).pages)
        compact="".join(text.split())
        if "511220" not in compact or "1:10" not in compact or "2023年1月13日" not in compact:
            raise ValueError("Official split identity/date/ratio not confirmed")
    with np.load(source/"dataset.npz",allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    meta=json.loads((ROOT/"output/reports/unified_v53/metadata.json").read_text())
    returns=pd.read_parquet(source/"research_returns_daily_38.parquet")
    original=returns.copy()
    old=pd.read_parquet(ROOT/"data_processed/asset_returns_daily.parquet")
    old.date=pd.to_datetime(old.date)
    level=old[old.asset_code.eq("credit_lgfv_aaa")].set_index("date").close
    day=pd.Timestamp("2023-01-16");previous=pd.Timestamp("2023-01-13")
    unadjusted=level.loc[day]/level.loc[previous]-1
    corrected=level.loc[day]*10/level.loc[previous]-1
    if not np.isclose(original.loc[day,"credit_lgfv_aaa"],unadjusted,atol=1e-6):
        raise ValueError("Split incident does not match frozen source")
    returns.loc[day,"credit_lgfv_aaa"]=corrected
    gap=pd.Timestamp("2026-05-15")
    if original.loc[gap,"credit_lgfv_aa"]>-.9:raise ValueError("Unknown basis incident changed")
    returns.loc[gap,"credit_lgfv_aa"]=np.nan
    if ((returns.abs()>.25)&returns.notna()).any().any():
        raise ValueError("Additional extreme move requires adjudication, not clipping")
    bars={k:pd.read_parquet(source/f"source_{k}.parquet") for k in ["open","high","low","close","vol","amount"]}
    a["returns"]=np.where(returns.notna(),returns,0).astype(np.float32)
    a["return_masks"]=returns.notna().to_numpy()
    a["local_values"],a["local_masks"],names=local_features(returns,bars)
    a["macro_values"],a["macro_masks"],a["regime_index"]=observable_grid(returns)
    sw=a["sleeve_reference_weights"]
    # The gap changes observed-at-t membership; use the same preferred-weight policy.
    from src.v45_data import reference_weights,label_reference_policy
    preferred,_=label_reference_policy(meta["assets"],meta["asset_modules"])
    sw=reference_weights(a["returns"],a["return_masks"],preferred,a["module_index"])
    a["sleeve_reference_weights"]=sw
    cb=pd.read_parquet(source/"cb_reference_returns.parquet").iloc[:,0].to_numpy()
    a.update(independent_cb_relative_labels(a["dates"],a["returns"],a["return_masks"],a["module_index"],sw,a["balanced_reference_weights"],cb))
    sr=np.sum(sw*np.where(a["return_masks"],a["returns"],0)[:,None,:],axis=-1)
    sm=(sw.sum(-1)>0)&((sw==0)|a["return_masks"][:,None,:]).all(-1)
    sr[:,2]=cb;sm[:,2]=np.isfinite(cb)
    for task,daily in [("asset",returns.to_numpy()),("sleeve",np.where(sm,sr,np.nan))]:
        raw_y=a[task+"_target"].copy();raw_m=a[task+"_target_mask"].copy()
        result=temporal_opportunity_targets(raw_y,raw_m,daily)
        a[task+"_raw_relative_target"],a[task+"_raw_relative_target_mask"]=raw_y,raw_m
        a[task+"_target"],a[task+"_target_mask"]=result["target"],result["mask"]
        a[task+"_opportunity_scale"],a[task+"_opportunity_offset"]=result["scale"],result["offset"]
        good=result["mask"]
        assert np.allclose(decode_opportunity(result["target"],result["scale"],result["offset"])[good],raw_y[good],atol=1e-6)
    validate_relative_dataset(a)
    assert len(meta["assets"])==38
    assert np.array_equal(a["dates"],pd.DatetimeIndex(returns.index).asi8)
    assert not a["return_masks"][list(returns.index).index(gap),meta["assets"].index("credit_lgfv_aa")]
    for h,j in [(5,0),(20,1),(60,2)]:
        assert not a["asset_target_mask"][-h:,j].any()
    np.savez_compressed(out/"dataset.npz",**a)
    returns.to_parquet(out/"research_returns_daily_38.parquet")
    manifest=copy.deepcopy(meta)
    manifest.update(version="v58_split_corrected_gap_preserved_research",local_features=names,
        prediction_units="risk_normalized_past_debiased_opportunity",nine_grid_preserved=True,
        parent_retired_dataset_sha256=sha(source/"dataset.npz"),
        source_manifest={**meta["source_manifest"],"split_evidence":dict(path=str(evidence.relative_to(ROOT)),sha256=sha(evidence)),
            "v58_builder":dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(Path(__file__)))},
        array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in a.items()})
    save(report/"metadata.json",manifest)
    incidents=[dict(asset="credit_lgfv_aaa",date=str(day.date()),raw_return=float(unadjusted),
                    corrected_return=float(corrected),method="verified1to10share_split",url=url),
               dict(asset="credit_lgfv_aa",date=str(gap.date()),raw_return=float(original.loc[gap,"credit_lgfv_aa"]),
                    corrected_return=None,method="unknown_index_basis_jump_masked_not_guessed")]
    save(report/"status.json",dict(status="rebuilt_not_trained",dataset_sha256=sha(out/"dataset.npz"),
        metadata_sha256=sha(report/"metadata.json"),incidents=incidents,
        assets=38,old_checkpoints_incompatible=True,formal_promotion=False,
        all_features_and_labels_rebuilt=True,remaining_cashflows_not_certified=True))
    print(json.dumps(incidents,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
