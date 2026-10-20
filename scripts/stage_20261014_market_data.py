"""Versioned raw acquisition, not automatic replacement of training inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data_raw/refresh_20261014_wind_verified"
CLI = Path.home() / ".agents/skills/wind-mcp-skill/scripts/cli.mjs"
START, END = "20260701", "20261014"
FATAL = {"KEY_MISSING", "KEY_INVALID", "KEY_FORBIDDEN_SERVER", "RATE_LIMIT_DAILY", "BALANCE_INSUFFICIENT"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def tables(value):
    if isinstance(value, str):
        try:
            yield from tables(json.loads(value))
        except (ValueError, TypeError):
            return
    elif isinstance(value, list):
        for item in value:
            yield from tables(item)
    elif isinstance(value, dict):
        if "columns" in value and "rows" in value:
            names = [c["name"] if isinstance(c, dict) else c for c in value["columns"]]
            yield pd.DataFrame(value["rows"], columns=names), value.get("unit", {})
        elif "date" in value and "indicatorInfo" in value:
            # Wind EDB uses a different schema from market bars. Keep source
            # metadata and missing values; a period date is NOT a release date.
            for indicator in value["indicatorInfo"]:
                if len(indicator.get("data", [])) != len(value["date"]):
                    raise ValueError("EDB date/value length mismatch")
                frame = pd.DataFrame({"period_date": value["date"], "value": indicator["data"]})
                for name in ["code", "name", "source", "unit", "calcFreq"]:
                    frame[name] = indicator.get(name)
                yield frame, {"value": indicator.get("unit")}
        else:
            for key in ["content", "text", "data", "result"]:
                if key in value:
                    yield from tables(value[key])


def call(tag, server, tool, params):
    path = DEST / "wind" / f"{tag}.json"
    request = dict(server=server, tool=tool, params=params)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["request"] != request:
            raise ValueError(f"Cached request mismatch: {path}")
        if record["exit_code"] == 0:
            return record["response"]
    for attempt in range(2):
        started = time.monotonic()
        try:
            proc = subprocess.run(["node", str(CLI), "call", server, tool, json.dumps(params, ensure_ascii=False)],
                                  capture_output=True, encoding="utf-8", errors="replace", timeout=75)
            response = json.loads(proc.stdout)
            error = response.get("error") or {} if isinstance(response, dict) else {}
            code = error.get("code") if isinstance(error, dict) else None
            record = dict(request=request, fetched_at=pd.Timestamp.now().isoformat(), response=response,
                          exit_code=proc.returncode, elapsed_seconds=time.monotonic()-started)
            save(path, record)
            if code in FATAL:
                raise PermissionError(f"Wind requires account repair: {code}")
            if proc.returncode != 0 or response.get("isError"):
                raise RuntimeError(f"Wind failed: {code or 'MCP_result_error'}")
            return response
        except PermissionError:
            raise
        except (subprocess.TimeoutExpired, RuntimeError, ValueError):
            if attempt:
                raise
            time.sleep(4)


def normalize_bars(response, start=START, end=END):
    found = list(tables(response))
    if len(found) != 1:
        raise ValueError("Expected exactly one bar table")
    frame, units = found[0]
    if not {"TIME", "MATCH"}.issubset(frame):
        raise ValueError("Missing bar schema")
    dates = frame.TIME.astype(str)
    # A domestic identifier silently resolving to an overseas stock is fatal
    # to this series, even when the API returns success.
    if dates.str.contains(r"[+-]\d\d:\d\d$").any() and not dates.str.endswith("+08:00").all():
        raise ValueError("Domestic series returned a non-China timezone")
    unit_text = json.dumps(units, ensure_ascii=False).upper()
    if any(x in unit_text for x in ["USD", "美元", "HKD", "港元"]):
        raise ValueError("Domestic series returned overseas currency")
    output = pd.DataFrame({"date": pd.to_datetime(dates.str[:10], errors="raise")})
    for source, target in [("OPEN", "open"), ("HIGH", "high"), ("LOW", "low"),
                           ("MATCH", "close"), ("VOLUME", "vol"), ("TURNOVER", "amount")]:
        output[target] = pd.to_numeric(frame[source], errors="coerce") if source in frame else np.nan
    if output.date.duplicated().any() or output.date.isna().any():
        raise ValueError("Duplicate/invalid market dates")
    if not output.date.between(pd.Timestamp(start), pd.Timestamp(end)).all():
        raise ValueError("Response exceeded requested cutoff")
    if output.empty or not np.isfinite(output.close).all() or not output.close.gt(0).all():
        raise ValueError("Empty/nonpositive/nonfinite closes")
    return output.sort_values("date").reset_index(drop=True)


def overlap_check(fresh, old):
    paired = fresh[["date", "close"]].merge(old[["date", "close"]], on="date", suffixes=("_new", "_old"))
    if len(paired) < 5:
        return dict(passed=False, reason="fewer_than_five_overlap_observations", overlap_rows=len(paired))
    errors = np.abs(paired.close_new/paired.close_old-1)
    error = float(errors.max())
    return dict(passed=error <= .001, reason="same_level_overlap" if error <= .001 else "price_basis_or_identifier_mismatch",
                overlap_rows=len(paired), max_relative_level_error=error)


def tasks():
    cfg = yaml.safe_load((ROOT / "config/asset_universe.yaml").read_text(encoding="utf-8"))
    result = []
    for spec in cfg["assets"]:
        if spec["code"] in ["alt_reits", "alt_us_bond"]:
            continue
        code = spec["ts_code"]
        # Vendor identifiers are separate. Validate each returned identity below.
        wind_code = code[:-3]+".CS" if code.endswith(".CB") else code
        if code == "H11098.CSI":
            wind_code = "H11098.SH"
        result.append(dict(asset=spec["code"], code=wind_code, source_code=code,
                           kind="fund" if spec["source_type"] == "fund_daily" else "index"))
    for code in ["180101.SZ", "180201.SZ", "180301.SZ", "180801.SZ", "508000.SH", "508001.SH", "508006.SH", "508027.SH", "508056.SH"]:
        result.append(dict(asset="reits_component_"+code, code=code, source_code=code, kind="fund"))
    for name, code in [("csi300", "000300.SH"), ("cb_index", "000832.CSI"), ("csi_dividend", "000922.CSI")]:
        result.append(dict(asset="feature_"+name, code=code, source_code=code, kind="index"))
    return result


def acquire_assets(status):
    existing = pd.read_parquet(ROOT / "data_processed/asset_returns_daily.parquet")
    existing["date"] = pd.to_datetime(existing.date)
    requests = tasks()
    frames = []
    for number, task in enumerate(requests, 1):
        name, code, kind = task["asset"], task["code"], task["kind"]
        tag = name+"_wind_sh" if code == "H11098.SH" else name
        entry = dict(dataset=name, requested_code=code, formal_replacement=False)
        if name.startswith("credit_") or name == "rate_30y":
            entry["semantic_status"] = "existing_proxy_only_not_strict_rating_industry_or_constant_30y"
        try:
            tool = "get_fund_info" if kind == "fund" else "get_index_basicinfo"
            identity = call(tag+"_identity", kind+"_data", tool,
                            dict(question=f"核实精确代码{code}的证券名称、Wind代码、币种和证券类型。不替换标的，无数据则明确返回。", lang="中文"))
            identity_tables = list(tables(identity))
            if not any(frame.astype(str).eq(code).any().any() for frame, _ in identity_tables):
                raise ValueError("Exact Wind identifier absent from identity response")
            response = call(tag+"_bars", kind+"_data", f"get_{kind}_kline",
                            dict(windcode=code, begin_date=START, end_date=END, period="10", aftype="0"))
            fresh = normalize_bars(response)
            fresh["asset_code"], fresh["wind_code"] = name, code
            old = existing.loc[existing.asset_code.eq(name)]
            check = overlap_check(fresh, old) if not old.empty else dict(passed=False, reason="raw_component_or_feature_requires_separate_assembly")
            entry.update(status="validated_candidate" if check["passed"] and fresh.date.max() == pd.Timestamp(END) else "quarantined_review",
                         rows=len(fresh), start=str(fresh.date.min().date()), end=str(fresh.date.max().date()), **check)
            # These are raw candidates, never a promoted return matrix.
            frames.append(fresh)
            fresh.to_parquet(DEST / "candidates" / f"{name}.parquet", index=False)
        except PermissionError:
            raise
        except Exception as exc:
            entry.update(status="blocked", error=str(exc))
        status.append(entry)
        save(DEST / "progress.json", dict(stage="wind_assets", done=number, total=len(requests), status=status))
        print(f"[assets {number}/{len(requests)}] {name}: {entry['status']}", flush=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(DEST / "candidates/all_market_bars.parquet", index=False)


def acquire_macro(status):
    queries = {
        "treasury_curve": "中国中债国债到期收益率1年、3年、5年、10年、30年，2026年7月1日至2026年10月14日，日频，单位百分比；保留原始指标名称。",
        "credit_curve": "中国中债中短期票据AAA到期收益率1年、3年、5年，2026年7月1日至2026年10月14日，日频，单位百分比。",
        "funding": "中国银行间质押式回购利率DR007、R007，2026年7月1日至2026年10月14日，日频，单位百分比。",
        "us10y": "美国国债10年期到期收益率，2026年7月1日至2026年10月14日，日频，单位百分比。",
        "inflation": "中国CPI同比、PPI同比，报告期2026年6月至9月，仅返回截至2026年10月14日已发布的数据；同时返回每期实际发布日期，未发布的不要填。",
        "activity": "中国官方制造业PMI，报告期2026年6月至9月，仅返回截至2026年10月14日已发布的数据；同时返回每期实际发布日期，未发布的不要填。",
        "liquidity": "中国M2同比、社会融资规模当月增量，报告期2026年6月至9月，仅返回截至2026年10月14日已发布的数据；同时返回每期实际发布日期及单位，未发布的不要填。",
    }
    for name, query in queries.items():
        try:
            result = call("macro_"+name, "economic_data", "get_economic_data", dict(
                metricIdsStr=query, beginDate="20260601" if name in {"inflation", "activity", "liquidity"} else START,
                endDate=END))
            count = sum(len(frame) for frame, _ in tables(result))
            for number, (frame, _) in enumerate(tables(result)):
                frame.to_parquet(DEST / "candidates" / f"macro_{name}_{number}.parquet", index=False)
            status.append(dict(dataset="macro_"+name, status="raw_requires_unit_and_release_audit" if count else "blocked_empty", rows=count, formal_replacement=False))
        except PermissionError:
            raise
        except Exception as exc:
            status.append(dict(dataset="macro_"+name, status="blocked", error=str(exc)))
        print(f"[macro] {name}: {status[-1]['status']}", flush=True)
        save(DEST / "progress.json", dict(stage="macro", status=status))


def acquire_cb(status):
    import tushare as ts
    pro = ts.pro_api(ts.get_token())
    calendar = pro.trade_cal(exchange="SSE", start_date=START, end_date=END, is_open="1")
    if calendar.empty:
        raise ValueError("Empty authoritative trading calendar")
    dates = sorted(calendar.cal_date.astype(str).unique())
    if dates[-1] != END:
        raise ValueError("Calendar does not reach requested completed session")
    folder = DEST / "tushare_cb"
    folder.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(folder / "calendar.parquet", index=False)
    for number, day in enumerate(dates, 1):
        path = folder / f"cb_daily_{day}.parquet"
        frame = pd.read_parquet(path) if path.exists() else pro.cb_daily(trade_date=day)
        if frame.empty or set(frame.trade_date.astype(str)) != {day} or frame.ts_code.duplicated().any():
            raise ValueError(f"Missing/invalid convertible daily panel: {day}")
        frame.to_parquet(path, index=False)
        print(f"[CB {number}/{len(dates)}] {day}: {len(frame)} bonds", flush=True)
        save(DEST / "progress.json", dict(stage="cb_daily", done=number, total=len(dates), latest=day))
        time.sleep(.25)
    basic = pro.cb_basic()
    basic.to_parquet(folder / "cb_basic_snapshot.parquet", index=False)
    status.append(dict(dataset="cb_daily", status="raw_closes_only_balance_and_membership_still_required",
                       days=len(dates), end=END, formal_replacement=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cb", action="store_true")
    args = parser.parse_args()
    (DEST / "candidates").mkdir(parents=True, exist_ok=True)
    status = []
    save(DEST / "registration.json", dict(start=START, end=END, created_at=pd.Timestamp.now().isoformat(),
        original_data_unchanged=True, new_models_trained=False, source_PIT_certified=False,
        domestic_vendor_crosswalk="iFinD .CB is not Wind .CS; require exact identity and overlap validation",
        plan="raw->identity/units/overlap/release audit->frozen model new-period evaluation->decide retraining"))
    try:
        acquire_assets(status)
        acquire_macro(status)
        if not args.skip_cb:
            acquire_cb(status)
    except BaseException as exc:
        save(DEST / "progress.json", dict(stage="failed", error=str(exc), status=status))
        raise
    pd.DataFrame(status).to_csv(DEST / "coverage_status.csv", index=False, encoding="utf-8-sig")
    hashes = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DEST.rglob("*") if p.is_file() and p.name not in {"manifest.json", "progress.json"}}
    save(DEST / "manifest.json", dict(files=hashes, status=status, training_ready=False, formal_replacement=False))
    save(DEST / "progress.json", dict(stage="acquisition_completed_not_training_ready", status=status))
    print("Raw acquisition complete; old data/models untouched. Not a complete training dataset.", flush=True)


if __name__ == "__main__":
    main()
