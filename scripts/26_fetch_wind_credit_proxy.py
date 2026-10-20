from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep_pipeline import build_feature_panels, build_monthly_returns, quality_tables, write_reports


SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")
START_DATE = "20180101"
END_DATE = "20260602"

WIND_CREDIT_PROXY = [
    {
        "asset_code": "credit_lgfv_aa_plus",
        "asset_name": "城投债AA+",
        "module": "credit",
        "sub_module": "credit_lgfv_aa_plus",
        "windcode": "931740.CSI",
        "source_type": "wind_index_kline",
        "data_source": "wind",
        "status": "ok",
        "note": "Wind补数：中证中诚信城投债1-3隐含AA+及以上指数，作为城投债AA+代理。",
    },
    {
        "asset_code": "credit_lgfv_aa",
        "asset_name": "城投债AA",
        "module": "credit",
        "sub_module": "credit_lgfv_aa",
        "windcode": "H11098.CSI",
        "source_type": "wind_index_kline",
        "data_source": "wind_proxy",
        "status": "ok",
        "note": "Wind补数代理：上证城投债指数。未精确拆AA评级，需Wind更细口径或中债估值曲线继续补齐。",
    },
    {
        "asset_code": "credit_ind_aa_plus",
        "asset_name": "产业债AA+",
        "module": "credit",
        "sub_module": "credit_ind_aa_plus",
        "windcode": "950127.CSI",
        "source_type": "wind_index_kline",
        "data_source": "wind_proxy",
        "status": "ok",
        "note": "Wind补数代理：沪信用债AA+指数。未精确拆产业债，需Wind更细口径继续补齐。",
    },
    {
        "asset_code": "credit_ind_aa",
        "asset_name": "产业债AA",
        "module": "credit",
        "sub_module": "credit_ind_aa",
        "windcode": "950128.CSI",
        "source_type": "wind_index_kline",
        "data_source": "wind_proxy",
        "status": "ok",
        "note": "Wind补数代理：沪信用债AA指数。未精确拆产业债，需Wind更细口径继续补齐。",
    },
]


def call_wind_index_kline(windcode: str, raw_path: Path | None = None) -> dict:
    if raw_path is not None and raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8"))
    params = {
        "windcode": windcode,
        "begin_date": START_DATE,
        "end_date": END_DATE,
        "period": "10",
    }
    cmd = [
        "node",
        str(SKILL_CLI),
        "call",
        "index_data",
        "get_index_kline",
        json.dumps(params, ensure_ascii=False, separators=(",", ":")),
    ]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180)
    if res.returncode != 0:
        raise RuntimeError(res.stdout.strip() or res.stderr.strip())
    outer = json.loads(res.stdout)
    text = outer["content"][0]["text"]
    payload = json.loads(text)
    if payload.get("error"):
        raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))
    return payload["data"]


def parse_wind_data(data: dict, asset: dict, fetch_time: str) -> pd.DataFrame:
    columns = [c["name"] for c in data.get("columns", [])]
    rows = data.get("rows", [])
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["_DATE"], format="%Y%m%d", errors="coerce")
    df["close"] = pd.to_numeric(df["MATCH"], errors="coerce")
    for src, dst in [("OPEN", "open"), ("HIGH", "high"), ("LOW", "low"), ("VOLUME", "vol"), ("TURNOVER", "amount")]:
        df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df["return_1d"] = df["close"].pct_change()
    df["asset_code"] = asset["asset_code"]
    df["code"] = asset["asset_code"]
    df["asset_name"] = asset["asset_name"]
    df["asset_class"] = asset["module"]
    df["module"] = asset["module"]
    df["sub_module"] = asset["sub_module"]
    df["ts_code"] = asset["windcode"]
    df["data_source"] = asset["data_source"]
    df["fetch_time"] = fetch_time
    keep = [
        "date",
        "asset_code",
        "code",
        "asset_name",
        "asset_class",
        "module",
        "sub_module",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "return_1d",
        "data_source",
        "fetch_time",
    ]
    return df[keep]


def main() -> None:
    raw_dir = ROOT / "data_raw" / "wind"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetch_time = datetime.now().isoformat(timespec="seconds")
    frames = []
    universe_rows = []
    status = {}

    for asset in WIND_CREDIT_PROXY:
        try:
            raw_path = raw_dir / f"index_kline_{asset['windcode']}_{START_DATE}_{END_DATE}.json"
            data = call_wind_index_kline(asset["windcode"], raw_path)
            raw_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            frame = parse_wind_data(data, asset, fetch_time)
            frames.append(frame)
            row = {
                **asset,
                "code": asset["asset_code"],
                "name": asset["asset_name"],
                "tradable_proxy": asset["windcode"],
                "start_date": frame["date"].min().date().isoformat() if not frame.empty else None,
                "end_date": frame["date"].max().date().isoformat() if not frame.empty else None,
                "n_obs": int(len(frame)),
                "fetch_time": fetch_time,
            }
            universe_rows.append(row)
            status[asset["asset_code"]] = f"ok:{len(frame)}"
        except Exception as exc:
            row = {
                **asset,
                "code": asset["asset_code"],
                "name": asset["asset_name"],
                "status": "wind_fetch_failed",
                "error": str(exc),
                "n_obs": 0,
                "fetch_time": fetch_time,
            }
            universe_rows.append(row)
            status[asset["asset_code"]] = f"failed:{exc}"

    add_daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    add_universe = pd.DataFrame(universe_rows)

    processed = ROOT / "data_processed"
    universe = pd.read_parquet(processed / "asset_universe.parquet")
    daily = pd.read_parquet(processed / "asset_returns_daily.parquet")
    replace_codes = [a["asset_code"] for a in WIND_CREDIT_PROXY]
    universe = universe[~universe["code"].isin(replace_codes) & ~universe["asset_code"].isin(replace_codes)]
    daily = daily[~daily["asset_code"].isin(replace_codes)]
    universe = pd.concat([universe, add_universe], ignore_index=True)
    if "asset_code" not in universe.columns:
        universe["asset_code"] = universe.get("code")
    universe["asset_code"] = universe["asset_code"].fillna(universe.get("code"))
    if "asset_name" not in universe.columns:
        universe["asset_name"] = universe.get("name")
    universe["asset_name"] = universe["asset_name"].fillna(universe.get("name"))
    if not add_daily.empty:
        daily = pd.concat([daily, add_daily], ignore_index=True).sort_values(["asset_code", "date"])

    monthly = build_monthly_returns(daily)
    fd, fm = build_feature_panels(daily, monthly)

    universe.to_parquet(processed / "asset_universe.parquet", index=False)
    daily.to_parquet(processed / "asset_returns_daily.parquet", index=False)
    monthly.to_parquet(processed / "asset_returns_monthly.parquet", index=False)
    fd.to_parquet(processed / "feature_panel_daily.parquet", index=False)
    fm.to_parquet(processed / "feature_panel_monthly.parquet", index=False)

    tables = quality_tables(universe, daily, monthly)
    write_reports(ProjectPathsShim(ROOT), tables, {f"wind_credit_proxy:{k}": v for k, v in status.items()})
    print(status)


class ProjectPathsShim:
    def __init__(self, root: Path):
        self.root = root

    @property
    def reports(self) -> Path:
        return self.root / "output" / "reports"


if __name__ == "__main__":
    main()
