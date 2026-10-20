"""Typed source reconciliation; current status and nontrading marks are not fills."""
import re
import numpy as np
import pandas as pd
from src.v49_point_in_time import next_session_after


def requested_codes(query):
    return re.findall(r"(?<!\d)\d{6}\.(?:SH|SZ)(?![A-Z])",query)


def clean_number(value):
    if isinstance(value,(int,float,np.number)):
        return float(value) if np.isfinite(value) else np.nan
    s=str(value).strip().replace(",","")
    if s in {"--","-","None","nan","","不适用"}:return np.nan
    m=re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(万|亿)?",s)
    if not m:return np.nan
    return float(m[1])*{"万":1e4,"亿":1e8,None:1}[m[2]]


def quote_table(frame,requested_codes,start,end):
    """Accept dated, code-identified yuan transaction-price table cells only."""
    columns=list(map(str,frame.columns))
    code=next((c for c in columns if c in {"证券代码","债券代码"}),None)
    date=next((c for c in columns if c in {"日期","交易日期","时间"}),None)
    prices=[c for c in columns if re.match(r"^(?:债券)?收盘价",c) and "元" in c and not any(x in c for x in ["估值","净价","全价","复权"])]
    previous=[c for c in columns if re.match(r"^(?:债券)?(?:前|昨)收盘价",c) and "元" in c]
    volume=next((c for c in columns if c.startswith("成交量")),None)
    if code is None or date is None or len(prices)!=1:return pd.DataFrame()
    out=pd.DataFrame({"ts_code":frame[code].astype(str).str.strip(),
        "date":pd.to_datetime(frame[date].astype(str).str.replace(r"\.0$","",regex=True),errors="coerce"),
        "close":frame[prices[0]].map(clean_number)})
    out["preclose"]=frame[previous[0]].map(clean_number) if len(previous)==1 else np.nan
    # Volume magnitude is not used as lots/shares: some endpoint metadata has
    # ambiguous units. Only a confirmed strictly positive observation is used.
    out["positive_volume"]=frame[volume].map(clean_number).gt(0) if volume else False
    out=out.loc[out.ts_code.isin(requested_codes)&out.date.between(pd.Timestamp(start),pd.Timestamp(end))]
    return out.loc[np.isfinite(out.close)&out.close.gt(0)].copy()


def issuance_registry(issues,master,dates):
    if issues.ts_code.duplicated().any():raise ValueError("Duplicate issuance registry")
    out=issues.set_index("ts_code").reindex(master.index).copy()
    for c in ["ann_date","res_ann_date"]:
        out[c]=pd.to_datetime(out[c].astype(str),format="%Y%m%d",errors="coerce")
    known_size=pd.to_numeric(master.issue_size,errors="coerce")
    raw=pd.to_numeric(out.issue_size,errors="coerce")
    # cb_issue's historical 100m-CNY values are serialized to four decimals;
    # their half-unit rounding error can be 5,000 CNY. This tolerance is a
    # source-precision contract, not a strategy-tuned reconciliation threshold.
    matched=pd.DataFrame({str(scale):np.isclose(raw*scale,known_size,rtol=1e-6,atol=max(1.,scale*.00005+1.))
                          & raw.gt(0)&known_size.gt(0) for scale in [1.,1e4,1e8]},index=out.index)
    out["unit_factor"]=np.nan
    for scale in [1.,1e4,1e8]:out.loc[matched[str(scale)]&matched.sum(axis=1).eq(1),"unit_factor"]=scale
    out["issue_size_yuan"]=raw*out.unit_factor
    out["size_rounding_error_yuan"]=out.issue_size_yuan-known_size
    types=out.issue_type.fillna("")
    out["public_method_verified"]=types.str.contains("网上|上网|网下")&~types.str.contains("定向|私募")
    out["result_chronology_valid"]=out.ann_date.notna()&out.res_ann_date.notna()&out.res_ann_date.ge(out.ann_date)
    out["result_known_at"]=next_session_after(out.res_ann_date,dates)
    conv=pd.to_datetime(master.conv_start_date.astype(str),errors="coerce")
    out["conversion_start"]=conv
    out["result_available_by_listing"]=out.result_known_at.le(master.list_date_parsed)
    out["issuance_record_usable"]=out.public_method_verified&out.result_chronology_valid&out.unit_factor.notna()
    return out


def fill_preconversion_known_balance(known,registry,dates):
    """No assumption of initial size once conversion may legally have begun."""
    out={k:v.copy() for k,v in known.items()}
    source=pd.DataFrame("missing",index=dates,columns=out["balance"].columns)
    source=source.mask(out["balance"].notna(),"dated_share_announcement")
    for code,row in registry.iterrows():
        if code not in out["balance"] or not row.issuance_record_usable:continue
        if pd.isna(row.result_known_at) or pd.isna(row.conversion_start):continue
        mask=(dates>=row.result_known_at)&(dates<row.conversion_start)&out["balance"][code].isna().to_numpy()
        out["balance"].loc[mask,code]=row.issue_size_yuan
        out["issue_size"].loc[mask,code]=row.issue_size_yuan
        source.loc[mask,code]="published_issue_result_before_contract_conversion"
    return out,source
