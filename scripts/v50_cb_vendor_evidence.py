"""Bounded, cached multi-vendor evidence collection; no fabricated market marks."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from io import StringIO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.stage_20261014_market_data import tables, save

DEST=ROOT/"data_raw/cb_evidence_v50"
IFIND=Path.home()/".agents/skills/ifind-finance-data/call-node.js"
WIND=Path.home()/".agents/skills/wind-mcp-skill/scripts/cli.mjs"


def vendor_request(vendor,tag,server,tool,params):
    path=DEST/vendor/f"{tag}.json"
    request=dict(server=server,tool=tool,params=params)
    if path.exists():
        record=json.loads(path.read_text(encoding="utf-8"))
        if record["request"]!=request:raise ValueError("Cached request mismatch")
        if record.get("exit_code")!=0:
            raise RuntimeError(f"{vendor} cached failure; see {path.name}; not retried")
        return record["response"]
    if vendor=="wind":
        args=["node",str(WIND),"call",server,tool,json.dumps(params,ensure_ascii=False)]
    elif vendor=="ifind":
        js="const {call}=require(process.argv[1]);call(process.argv[2],process.argv[3],JSON.parse(process.argv[4])).then(x=>console.log(JSON.stringify(x))).catch(e=>{console.error(e.message);process.exitCode=1});"
        args=["node","-e",js,str(IFIND),server,tool,json.dumps(params,ensure_ascii=False)]
    else:raise ValueError("Unknown vendor")
    start=time.monotonic()
    proc=subprocess.run(args,capture_output=True,encoding="utf-8",errors="replace",timeout=90)
    try:response=json.loads(proc.stdout)
    except ValueError:response=dict(parse_error=True,stdout=proc.stdout[:2000])
    save(path,dict(request=request,response=response,exit_code=proc.returncode,
                   fetched_at=pd.Timestamp.now(tz="UTC").isoformat(),elapsed_seconds=time.monotonic()-start))
    if proc.returncode!=0 or (vendor=="ifind" and not response.get("ok",False)):
        raise RuntimeError(f"{vendor} request failed; see {path.name}")
    return response


def ifind_tables(value):
    """Only structured tables; never extract prices from generated prose."""
    if isinstance(value,str):
        try:yield from ifind_tables(json.loads(value))
        except (ValueError,TypeError):
            # Some bond endpoints return explicit tabular Markdown rather
            # than datas[]. Parse tables with a real Markdown/HTML parser;
            # no figures are extracted from surrounding narrative.
            from markdown_it import MarkdownIt
            html=MarkdownIt("default").render(value.replace("\\n","\n"))
            if "<table>" in html:
                for f in pd.read_html(StringIO(html)):
                    yield f,dict(transport="explicit_markdown_table_no_prose_extraction")
            return
    elif isinstance(value,list):
        for item in value:yield from ifind_tables(item)
    elif isinstance(value,dict):
        if "columns" in value and "data" in value and isinstance(value["data"],list):
            names=[c["name"] if isinstance(c,dict) else c for c in value["columns"]]
            yield pd.DataFrame(value["data"],columns=names),value.get("attrs",{})
        else:
            for key in ["data","result","content","text","datas","answer"]:
                if key in value:yield from ifind_tables(value[key])


def acquire_remaining():
    missing=pd.read_csv(ROOT/"output/reports/cb_rebuild_20261014_v49/exit_aware_repaired/remaining_security_days.csv",parse_dates=["date"])
    tasks=[]
    for day,g in missing.groupby("date",sort=True):
        for i in range(0,len(g),4):
            part=g.iloc[i:i+4]
            names="、".join(part.ts_code+part.bond_short_name)
            start=(day-pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end=(day+pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            query=f"{names}，{start}至{end}每日原始交易所收盘价、前收盘价、成交量（明确单位）和当日是否停牌。必须逐行返回日期、精确证券代码和价格。不要估值净价替代成交价，不要用今日交易状态代替历史状态，不要插值填价，非交易日或无交易明确标注。"
            tasks.append((day,i,query))
    for n,(day,i,query) in enumerate(tasks,1):
        tag=f"remaining_{day:%Y%m%d}_{i}"
        try:
            response=vendor_request("ifind",tag,"bond","bond_market_data",dict(query=query))
            count=sum(len(f) for f,_ in ifind_tables(response))
            print(f"iFinD {n}/{len(tasks)} {tag}: {count} table rows",flush=True)
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            print(f"iFinD {tag}: {exc}",flush=True)
        save(ROOT/"output/reports/cb_evidence_v50/quote_progress.json",dict(status="running",completed=n,total=len(tasks)))
    save(ROOT/"output/reports/cb_evidence_v50/quote_progress.json",dict(status="completed_raw_evidence",completed=len(tasks),total=len(tasks)))


def probe():
    queries=[
        ("wind","march28","bond_data","get_bond_market_data",dict(question="精确查询113050.SH南银转债、113065.SH齐鲁转债、110070.SH凌钢转债、113610.SH灵康转债，2025年3月27日至3月31日每日交易所收盘价、前收盘价、成交量及停复牌状态。必须返回日期、证券代码和原始数据，不要估值净价替代成交收盘价，无交易的明确区分。",lang="中文")),
        ("ifind","march28","bond","bond_market_data",dict(query="113050.SH南银转债、113065.SH齐鲁转债、110070.SH凌钢转债、113610.SH灵康转债，2025年3月27日至3月31日每日收盘价、前收盘价、成交量和停牌状态。必须返回精确证券代码和日期；交易所成交价，不能用估值净价或插值填补。")),
        ("ifind","default_exit","news","search_notice",dict(query="蓝盾转债123015在2023年7月、搜特转债128100在2023年8月终止上市或退市整理交易，最后交易日和公告日期",time_start="2023-06-01",time_end="2023-08-31",size=8)),
    ]
    for vendor,tag,server,tool,params in queries:
        try:
            result=vendor_request(vendor,tag,server,tool,params)
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            print(str(exc),flush=True)
            continue
        tab=list(tables(result) if vendor=="wind" else ifind_tables(result))
        print(vendor,tag,"tables",len(tab),flush=True)
        for f,attrs in tab:
            print(f.head(6).to_string(index=False),flush=True)
            print("metadata",str(attrs)[:600],flush=True)
        if not tab:print(str(result)[:3500],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--probe",action="store_true")
    parser.add_argument("--remaining",action="store_true")
    args=parser.parse_args()
    if args.probe:probe()
    if args.remaining:acquire_remaining()
    if not (args.probe or args.remaining):parser.error("Use --probe or --remaining")
