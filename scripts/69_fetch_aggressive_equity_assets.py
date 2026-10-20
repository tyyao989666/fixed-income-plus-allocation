from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep_pipeline import (  # noqa: E402
    build_monthly_returns,
    ensure_dirs,
    fetch_with_cache,
    get_tushare_pro,
    load_yaml,
    normalize_trade_frame,
)


ASSET_CODES = {"equity_chinext", "equity_csi1000"}


def merge_by_asset_code(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming.empty:
        return existing
    if existing.empty:
        return incoming
    drop_codes = set(incoming["asset_code"].dropna().astype(str).unique())
    kept = existing[~existing["asset_code"].astype(str).isin(drop_codes)].copy()
    out = pd.concat([kept, incoming], ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date", "asset_code"]).sort_values(["asset_code", "date"]).reset_index(drop=True)


def main() -> None:
    paths = ensure_dirs(ROOT)
    cfg = load_yaml(paths.config / "asset_universe.yaml")
    start_date = str(cfg["project"]["start_date"])
    end_date = str(cfg["project"]["end_date"])
    sleep_seconds = float(cfg["project"].get("sleep_seconds", 0.08))
    fetch_time = datetime.now().isoformat(timespec="seconds")
    pro = get_tushare_pro()

    frames: list[pd.DataFrame] = []
    universe_rows: list[dict] = []
    for asset in cfg.get("assets", []):
        if str(asset.get("code")) not in ASSET_CODES:
            continue
        endpoint = asset.get("source_type", "fund_daily")
        df = fetch_with_cache(
            paths,
            pro,
            endpoint,
            f"{asset.get('ts_code')}_{start_date}_{end_date}",
            {"ts_code": asset.get("ts_code"), "start_date": start_date, "end_date": end_date},
            sleep_seconds,
        )
        norm = normalize_trade_frame(df, asset, fetch_time)
        row = dict(asset)
        row["asset_code"] = asset["code"]
        row["fetch_time"] = fetch_time
        if norm.empty:
            row.update({"status": "empty", "start_date": None, "end_date": None, "n_obs": 0})
        else:
            row.update(
                {
                    "status": "ok",
                    "start_date": norm["date"].min().date().isoformat(),
                    "end_date": norm["date"].max().date().isoformat(),
                    "n_obs": int(len(norm)),
                }
            )
            frames.append(norm)
        universe_rows.append(row)

    incoming_daily = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if incoming_daily.empty:
        raise RuntimeError("No aggressive equity asset data fetched.")

    daily_path = paths.processed / "asset_returns_daily.parquet"
    monthly_path = paths.processed / "asset_returns_monthly.parquet"
    universe_path = paths.processed / "asset_universe.parquet"

    existing_daily = pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame()
    merged_daily = merge_by_asset_code(existing_daily, incoming_daily)
    merged_monthly = build_monthly_returns(merged_daily)

    existing_universe = pd.read_parquet(universe_path) if universe_path.exists() else pd.DataFrame()
    incoming_universe = pd.DataFrame(universe_rows)
    if not existing_universe.empty:
        existing_universe = existing_universe[~existing_universe["asset_code"].astype(str).isin(ASSET_CODES)].copy()
    merged_universe = pd.concat([existing_universe, incoming_universe], ignore_index=True, sort=False)

    merged_daily.to_parquet(daily_path, index=False)
    merged_monthly.to_parquet(monthly_path, index=False)
    merged_universe.to_parquet(universe_path, index=False)

    print(
        {
            "daily_rows": len(merged_daily),
            "monthly_rows": len(merged_monthly),
            "added_assets": sorted(ASSET_CODES),
            "fetched_rows": {r["code"]: r.get("n_obs", 0) for r in universe_rows},
            "monthly_end": str(pd.to_datetime(merged_monthly["date"]).max().date()),
        }
    )


if __name__ == "__main__":
    main()
