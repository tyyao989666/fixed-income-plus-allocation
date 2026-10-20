"""Bounded, cached exact-identity acquisition; never writes existing datasets."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage_20261014_market_data import tables, normalize_bars

RAW = ROOT / "data_raw/credit_identity_v73"
OUT = ROOT / "output/reports/credit_identity_v73"
CLI = Path.home() / ".agents/skills/wind-mcp-skill/scripts/cli.mjs"
CODE = "H11098.SH"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def call(tag, tool, params):
    path = RAW / f"{tag}.json"
    request = dict(server="index_data", tool=tool, params=params)
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["request"] != request:
            raise ValueError("Cached request identity mismatch")
    else:
        start = time.monotonic()
        try:
            proc = subprocess.run(["node", str(CLI), "call", "index_data", tool,
                                   json.dumps(params, ensure_ascii=False)],
                                  capture_output=True, encoding="utf-8", errors="replace", timeout=80)
            result = dict(request=request, response=json.loads(proc.stdout), exit_code=proc.returncode,
                          fetched_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                          elapsed_seconds=time.monotonic() - start)
        except subprocess.TimeoutExpired:
            result = dict(request=request, response={}, exit_code=-1, failure="timeout_80s_no_retry")
        save(path, result)
    error = result.get("response", {}).get("error") or {}
    if result["exit_code"] != 0 or result.get("response", {}).get("isError"):
        code = error.get("code", "vendor_or_timeout_error")
        if code == "KEY_MISSING":
            subprocess.run(["node", str(CLI), "open-portal"], capture_output=True, timeout=20)
        raise RuntimeError("Wind request failed: " + code)
    return result["response"]


def bars(tag, start, end):
    result = call(tag, "get_index_kline", dict(windcode=CODE, begin_date=start, end_date=end, period="10"))
    frame = normalize_bars(result, start=start, end=end)
    # Sanity exclusion only: plausible magnitude does not itself certify identity.
    if frame.close.gt(10000).any():
        raise ValueError("Bond index request returned incompatible five-digit price levels")
    return frame


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "status.json", dict(status="running_exact_identity_probe", formal_replacement=False))
    try:
        result = call("identity", "get_index_basicinfo", dict(
            question="核实精确代码H11098.SH的证券全称、Wind代码、币种、发布机构和指数类型。不替换标的，无数据明确返回。", lang="中文"))
        parts = [f for f, _ in tables(result)]
        flattened = " ".join(f.to_string(index=False) for f in parts)
        if CODE not in flattened or "城投债" not in flattened or "CNY" not in flattened:
            raise ValueError("Exact identity/CNY not confirmed")
        print("V73 exact identity confirmed; checking 2018 and overlap probes", flush=True)
        early = bars("probe_2018", "20180101", "20180119")
        if len(early) < 5:
            raise ValueError("Incomplete historical probe")
        overlap = bars("probe_overlap", "20260701", "20260721")
        registered = pd.read_parquet(ROOT / "data_raw/refresh_20261014_wind_verified/candidates/credit_lgfv_aa.parquet")
        pairs = overlap.merge(registered[["date", "close"]], on="date", suffixes=("_new", "_verified"))
        if len(pairs) < 5 or not np.allclose(pairs.close_new, pairs.close_verified, rtol=.001, atol=0):
            raise ValueError("Historical request disagrees with verified recent index basis")
        frames = []
        for year in range(2018, 2027):
            stop = "20261014" if year == 2026 else f"{year}1231"
            frame = bars(f"history_{year}", f"{year}0101", stop)
            if len(frame) < (150 if year == 2026 else 220):
                raise ValueError(f"Possibly incomplete annual history: {year}")
            frames.append(frame)
            print(f"V73 annual history {year}: {len(frame)} rows", flush=True)
            save(OUT / "status.json", dict(status="running_annual_download", completed_year=year, formal_replacement=False))
        joined = pd.concat(frames, ignore_index=True).sort_values("date")
        if joined.date.duplicated().any():
            raise ValueError("Duplicate history dates")
        # Full history and both probes must agree; never fit scale at a splice.
        for probe in (early, overlap):
            pair = joined.merge(probe[["date", "close"]], on="date", suffixes=("_history", "_probe"))
            if len(pair) != len(probe) or not np.allclose(pair.close_history, pair.close_probe, rtol=1e-8, atol=1e-8):
                raise ValueError("Annual history inconsistent with independent probe")
        joined["source_identifier"] = CODE
        joined.to_parquet(RAW / "correct_credit_index_levels.parquet", index=False)
        save(OUT / "status.json", dict(status="exact_history_candidate_downloaded_not_activated",
            rows=len(joined), start=str(joined.date.min().date()), end=str(joined.date.max().date()),
            exact_identity_verified=True, recent_overlap_rows=len(pairs), full_total_return_definition_certified=False,
            strict_AA_rating_universe=False, formal_replacement=False, models_retrained=False,
            source="Wind financial data service"))
    except Exception as exc:
        save(OUT / "status.json", dict(status="blocked_no_replacement", error=str(exc),
            formal_replacement=False, models_retrained=False, original_datasets_untouched=True))
        print("V73 blocked: " + str(exc), flush=True)
        return 1
    finally:
        save(OUT / "raw_hashes.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(RAW.glob("*")) if p.is_file()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
