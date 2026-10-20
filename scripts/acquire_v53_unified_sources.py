"""Cached refresh of complete REIT histories, US curves and CB coupon contracts."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import tushare as ts
from scripts.run_v52_execution_preflight import save, sha

DEST = ROOT / "data_raw/unified_v53"
REITS = ["180101.SZ", "180201.SZ", "180301.SZ", "180801.SZ", "508000.SH",
         "508001.SH", "508006.SH", "508027.SH", "508056.SH"]


def fetch(pro, endpoint, tag, **params):
    path = DEST / (tag + ".parquet")
    request = dict(endpoint=endpoint, params=params)
    if path.exists() and path.with_suffix(".json").exists():
        saved = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if saved["request"] != request:
            raise ValueError("Raw cache request mismatch")
        return pd.read_parquet(path)
    frame = getattr(pro, endpoint)(**params)
    if frame is None or len(frame) >= 2000:
        raise ValueError(f"Absent or possibly capped response: {tag}")
    frame.to_parquet(path, index=False)
    save(path.with_suffix(".json"), dict(request=request, rows=len(frame),
        fetched_at=pd.Timestamp.now(tz="UTC").isoformat(), sha256=sha(path)))
    time.sleep(.15)
    return frame


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    progress = DEST / "progress.json"
    pro = ts.pro_api(ts.get_token(), timeout=30)
    tasks, errors = 0, []
    save(progress, dict(status="running", completed=0))
    # Split annual histories; no truncated 2000-row full-history request.
    for code in REITS:
        for year in range(2021, 2027):
            start, end = f"{year}0101", min(f"{year}1231", "20261014")
            fetch(pro, "fund_daily", f"reit_{code}_{year}", ts_code=code, start_date=start, end_date=end)
            tasks += 1
        fetch(pro, "fund_div", f"reit_div_{code}", ts_code=code)
        tasks += 1
        save(progress, dict(status="running", stage="reits", completed=tasks, current=code))
        print(f"V53 REIT {code}: annual histories and distributions cached", flush=True)
    identity = fetch(pro, "index_basic", "nanhua_identity", market="NH")
    for asset, code, expected in [("alt_commodity", "NHCI.NH", "南华商品指数"),
                                   ("alt_chemical", "NHECI.NH", "南华能化指数"),
                                   ("alt_nonferrous", "NHNFI.NH", "南华有色金属")]:
        selected = identity.loc[identity.ts_code.eq(code)]
        if len(selected) != 1 or selected.iloc[0]["name"] != expected:
            raise ValueError("Nanhua code identity mismatch")
        fetch(pro, "index_daily", f"nanhua_gap_{asset}", ts_code=code, start_date="20231227", end_date="20240112")
        tasks += 1
        print(f"V53 exact Nanhua missing-session probe {asset}: cached", flush=True)
    for year in range(2017, 2027):
        frame = fetch(pro, "us_tycr", f"us_curve_{year}", start_date=f"{year}0101",
                      end_date=min(f"{year}1231", "20261014"))
        if not {"date", "y10"}.issubset(frame) or frame.empty:
            raise ValueError("US Treasury curve identity/schema missing")
        tasks += 1
        print(f"V53 US Treasury {year}: {len(frame)} rows", flush=True)
    for year in range(2017, 2027):
        frame = fetch(pro, "index_daily", f"cb_reference_index_{year}", ts_code="000832.CSI",
            start_date=f"{year}0101", end_date=min(f"{year}1231", "20261014"))
        if frame.empty or not frame.ts_code.eq("000832.CSI").all():
            raise ValueError("Independent CB reference identity missing")
        tasks += 1
        print(f"V53 CB reference 000832.CSI {year}: {len(frame)} rows", flush=True)
    from scripts.v50_cb_vendor_evidence import vendor_request, ifind_tables
    for year in range(2017, 2027):
        tag = f"v53_usdcny_{year}"
        query = (f"中国人民银行人民币汇率中间价：美元兑人民币，{year}-01-01至{min(str(year)+'-12-31', '2026-10-14')}，"
                 "每日原始数值，单位人民币元/1美元；返回精确指标名称、来源、日期和值，排除离岸人民币CNH、预测和移动平均。")
        # Preserve the already-cached initial request wording exactly.
        response = vendor_request("ifind", tag, "edb", "get_edb_data", dict(query=query))
        field = "中间价:美元兑人民币"
        matches = [f for f, attrs in ifind_tables(response) if field in f and
                   attrs.get(field, {}).get("data_source") == "中国外汇交易中心" and
                   attrs.get(field, {}).get("index_id") == "M002842089"]
        if len(matches) == 0:
            # Empty natural-language lookup is not numeric evidence. Retry a
            # bounded split using the exact already-identified EDB series ID.
            for half, (left, right) in enumerate([(f"{year}-01-01", f"{year}-06-30"),
                                                     (f"{year}-07-01", min(f"{year}-12-31", "2026-10-14"))]):
                query = f"指标M002842089（中间价:美元兑人民币，中国外汇交易中心），{left}至{right}每日原始数值，返回结构化日期和值，不要换指标或预测。"
                retry = vendor_request("ifind", tag + f"_exact_half{half}", "edb", "get_edb_data", dict(query=query))
                matches.extend([f for f, attrs in ifind_tables(retry) if field in f and
                    attrs.get(field, {}).get("data_source") == "中国外汇交易中心" and
                    attrs.get(field, {}).get("index_id") == "M002842089"])
            if len(matches) == 2:
                matches = [pd.concat(matches, ignore_index=True)]
        if len(matches) != 1:
            raise ValueError("FX series exact identity/source not unique")
        f = matches[0]
        fx = pd.DataFrame(dict(source_date=pd.to_datetime(f["日期"], errors="raise"),
                               cny_per_usd=pd.to_numeric(f[field], errors="raise")))
        if fx.source_date.duplicated().any() or not fx.source_date.between(pd.Timestamp(year,1,1), pd.Timestamp(min(str(year)+'1231','20261014'))).all():
            raise ValueError("FX observation chronology mismatch")
        fx.to_parquet(DEST / f"usdcny_parity_{year}.parquet", index=False)
        tasks += 1
        print(f"V53 CNY/USD central parity {year}: {len(fx)} observations", flush=True)
    master = pd.read_parquet(ROOT / "data_processed/cb_evidence_v50/historical_public_universe.parquet")
    codes = list(master.index)
    for i in range(0, len(codes), 40):
        group = codes[i:i+40]
        frame = fetch(pro, "cb_rate", f"cb_coupon_contract_{i:04d}", ts_code=",".join(group),
                      fields="ts_code,rate_freq,rate_start_date,rate_end_date,coupon_rate")
        if len(frame) and not frame.ts_code.isin(group).all():
            raise ValueError("Coupon response contains unrequested security")
        tasks += 1
        save(progress, dict(status="running", stage="coupon_contracts", completed=tasks, codes=min(i+40,len(codes))))
        print(f"V53 coupon CONTRACTS {min(i+40,len(codes))}/{len(codes)}; not payment evidence", flush=True)
    save(progress, dict(status="completed", completed=tasks, errors=errors,
        cutoff="2026-10-14", coupon_contracts_are_not_actual_payment_events=True,
        source_manifest={p.name: sha(p) for p in DEST.glob("*.parquet")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(DEST / "progress.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
