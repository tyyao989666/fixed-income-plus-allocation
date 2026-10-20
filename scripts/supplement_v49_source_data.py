"""Fetch dated CB disclosures/dividends and exact single macro requests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import stage_20261014_market_data as stage

BASE = stage.DEST
DEST = BASE / "supplement_v49"
END = "20261014"


def fetch(pro, endpoint, path, **params):
    if path.exists():
        return pd.read_parquet(path)
    frame = getattr(pro, endpoint)(**params)
    if frame is None:
        raise ValueError(f"No response: {endpoint}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def share_batch(pro, codes, number):
    path = DEST / "tushare" / f"cb_share_{number}.parquet"
    frame = fetch(pro, "cb_share", path, ts_code=",".join(codes), start_date="20260101", end_date=END)
    if len(frame) >= 2000:
        if len(codes) == 1:
            raise ValueError(f"Single CB response possibly truncated: {codes[0]}")
        middle = len(codes)//2
        return pd.concat([share_batch(pro, codes[:middle], str(number)+"a"),
                          share_batch(pro, codes[middle:], str(number)+"b")], ignore_index=True)
    if not frame.empty and not set(frame.ts_code).issubset(codes):
        raise ValueError("Unexpected security in disclosure response")
    return frame


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    (DEST / "candidates").mkdir(parents=True, exist_ok=True)
    status = []
    stage.save(DEST / "progress.json", dict(status="running", stage="CB_disclosures"))
    pro = ts.pro_api(ts.get_token())
    raw = pd.concat([pd.read_parquet(p, columns=["ts_code"]) for p in (BASE / "tushare_cb").glob("cb_daily_*.parquet")])
    codes = sorted(raw.ts_code.drop_duplicates())
    shares = []
    for left in range(0, len(codes), 20):
        shares.append(share_batch(pro, codes[left:left+20], left//20))
        stage.save(DEST / "progress.json", dict(status="running", stage="CB_disclosures", done=min(left+20,len(codes)), total=len(codes)))
        print(f"[CB disclosures] {min(left+20,len(codes))}/{len(codes)}", flush=True)
        time.sleep(.3)
    frame = pd.concat(shares, ignore_index=True).drop_duplicates()
    for column in ["publish_date", "end_date"]:
        frame[column] = pd.to_datetime(frame[column].astype(str), format="%Y%m%d", errors="coerce")
    valid = frame.publish_date.notna() & frame.end_date.notna() & frame.end_date.le(frame.publish_date) & frame.publish_date.le(pd.Timestamp(END))
    frame["chronology_valid"] = valid
    frame.to_parquet(DEST / "candidates/cb_share_dated_disclosures.parquet", index=False)
    # Conflicting records on the same release and period cannot be silently
    # resolved by row order; keep them for adjudication.
    conflicts = frame.loc[frame.duplicated(["ts_code", "publish_date", "end_date"], keep=False)]
    conflicts.to_parquet(DEST / "candidates/cb_share_conflicts.parquet", index=False)
    status.append(dict(dataset="CB_disclosures", rows=len(frame), securities=frame.ts_code.nunique(),
                       invalid_chronology=int((~valid).sum()), conflicting_rows=len(conflicts),
                       role="announcement_asof_balance_not_daily_true_balance", ready=False))
    calls = fetch(pro, "cb_call", DEST / "tushare/cb_calls.parquet", start_date="20260101", end_date=END)
    issues = fetch(pro, "cb_issue", DEST / "tushare/cb_issues.parquet", start_date="20260101", end_date=END)
    status.extend([dict(dataset="CB_calls", rows=len(calls), ready=False), dict(dataset="CB_issues", rows=len(issues), ready=False)])
    for code in ["180101.SZ", "180201.SZ", "180301.SZ", "180801.SZ", "508000.SH", "508001.SH", "508006.SH", "508027.SH", "508056.SH"]:
        bars = fetch(pro, "fund_daily", DEST / "tushare" / f"reits_unadjusted_{code}.parquet",
                     ts_code=code, start_date="20260701", end_date=END)
        dividend = fetch(pro, "fund_div", DEST / "tushare" / f"div_{code}.parquet", ts_code=code)
        unique = dividend.drop_duplicates()
        # No sum before deduplication: providers may return identical rows.
        unique.to_parquet(DEST / "candidates" / f"div_{code}_deduplicated.parquet", index=False)
        status.append(dict(dataset="dividend_"+code, rows=len(dividend), unique_rows=len(unique),
                           unadjusted_bar_rows=len(bars), ready=False))
        print(f"[dividend] {code}: {len(dividend)} -> {len(unique)}", flush=True)
    stage.DEST = DEST
    macros = {
        "cpi_exact": "中国:CPI:当月同比，来源国家统计局，排除国际货币基金组织、IMF、向前复制、移动平均和预测值",
        "ppi_exact": "中国:PPI:当月同比，来源国家统计局，排除向前复制、移动平均和预测值",
        "pmi_exact": "中国:制造业PMI，来源国家统计局，只要总体指数，排除分项、企业规模分项、向前复制和预测值",
        "dr007_exact": "DR007，来源中国货币网，只要银行间存款类机构质押式回购加权平均利率，排除移动平均、向前复制、成交量和交易所利率",
        "r007_exact": "中国:银行间质押式回购加权利率:7天，来源中国货币网，只要原始每日利率，排除交易所、向前复制和移动平均",
    }
    for name, query in macros.items():
        try:
            response = stage.call(name, "economic_data", "get_economic_data", dict(
                metricIdsStr=query, beginDate="20260601", endDate=END, searchType="1"))
            parts = list(stage.tables(response))
            for i, (part, _) in enumerate(parts):
                part.to_parquet(DEST / "candidates" / f"{name}_{i}.parquet", index=False)
            status.append(dict(dataset=name, tables=len(parts), ready=False, reason="raw_requires_exact_source_and_release_verification"))
        except PermissionError:
            raise
        except Exception as exc:
            status.append(dict(dataset=name, ready=False, error=str(exc)))
        print(f"[macro] {name}: captured", flush=True)
        stage.save(DEST / "progress.json", dict(status="running", stage="macro", datasets=status))
    # Use iFinD only for explicitly missing ChinaBond curves, not as a way to
    # bypass a Wind key, rate limit or account error.
    cli = Path.home()/".agents/skills/ifind-finance-data/call-node.js"
    js = "const {call}=require(process.argv[1]);call('edb','get_edb_data',JSON.parse(process.argv[2])).then(x=>console.log(JSON.stringify(x))).catch(e=>{console.error(e.message);process.exitCode=1});"
    for name, query in {
        "treasury30": "中债国债到期收益率:30年，2026-07-01至2026-10-14，日频，原始值，非拟合或代理值",
        "treasury10": "中债国债到期收益率:10年，2026-07-01至2026-10-14，日频，原始值，非拟合或代理值",
        "creditAAA3": "中债中短期票据收益率曲线(AAA):3年，2026-07-01至2026-10-14，日频，原始值",
    }.items():
        path = DEST / "ifind" / f"{name}.json"
        if not path.exists():
            try:
                proc = subprocess.run(["node", "-e", js, str(cli), json.dumps(dict(query=query), ensure_ascii=False)],
                                      capture_output=True, encoding="utf-8", errors="replace", timeout=75)
                response = json.loads(proc.stdout)
                stage.save(path, dict(query=query, fetched_at=pd.Timestamp.now().isoformat(), response=response, exit_code=proc.returncode))
            except Exception as exc:
                stage.save(path, dict(query=query, error=str(exc)))
        status.append(dict(dataset="ifind_"+name, ready=False, reason="raw_response_requires_series_identity_and_unit_audit"))
        print(f"[curve] {name}: captured", flush=True)
    hashes = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DEST.rglob("*") if p.is_file() and p.name not in {"manifest.json", "progress.json"}}
    stage.save(DEST / "manifest.json", dict(status=status, hashes=hashes, training_ready=False, formal_replacement=False))
    stage.save(DEST / "progress.json", dict(status="completed_raw_not_training_ready", datasets=status))
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        stage.save(DEST / "progress.json", dict(status="failed", error=str(exc), training_ready=False))
        raise
