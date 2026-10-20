from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market_skill import MarketSkillPaths, build_ep_views  # noqa: E402


PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"

ASSET_VIEW_MODULES = {
    "rates": ["rate_1_3y", "rate_3_5y", "rate_7_10y", "rate_30y"],
    "credit": [
        "credit_lgfv_aaa",
        "credit_lgfv_aa_plus",
        "credit_lgfv_aa",
        "credit_ind_aaa",
        "credit_ind_aa_plus",
        "credit_ind_aa",
    ],
    "equity": [
        "equity_broad",
        "equity_chinext",
        "equity_csi1000",
        "equity_tech_growth",
        "equity_dividend_utility",
        "equity_manufacturing",
        "equity_consumption",
        "equity_healthcare",
        "equity_fin_property",
        "equity_cyclical",
    ],
    "alternatives": ["alt_gold", "alt_commodity", "alt_nonferrous", "alt_chemical", "alt_us_bond", "alt_reits"],
}


def _read_success_dates(audit_path: Path) -> set[pd.Timestamp]:
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing audit file: {audit_path}")
    dates: set[pd.Timestamp] = set()
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if bool(item.get("fallback")) or not item.get("date"):
            continue
        dates.add(pd.Timestamp(item["date"]).normalize())
    return dates


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"])


def _write_ep(raw_views: pd.DataFrame, raw_path: Path, ep_path: Path) -> int:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_views = raw_views.sort_values(["date", "validator_status", "source_asset", "target_asset"]).reset_index(drop=True)
    raw_views.to_parquet(raw_path, index=False)
    ep = build_ep_views(raw_views, paths=MarketSkillPaths(ep_views=ep_path))
    return len(ep)


def _score_assets(history: pd.DataFrame, codes: list[str]) -> pd.Series:
    rows = []
    for code in codes:
        series = history[history["asset_code"].eq(code)].sort_values("date")["return_1d"].dropna().astype(float)
        if len(series) < 40:
            continue
        ret20 = float((1.0 + series.tail(20)).prod() - 1.0)
        ret60 = float((1.0 + series.tail(60)).prod() - 1.0) if len(series) >= 60 else ret20
        vol20 = float(series.tail(20).std(ddof=0))
        rows.append({"asset_code": code, "raw_score": 0.65 * ret20 + 0.35 * ret60 - 0.15 * vol20})
    if len(rows) < 2:
        return pd.Series(dtype=float)
    score = pd.DataFrame(rows).set_index("asset_code")["raw_score"]
    centered = score - float(score.mean())
    denom = float(centered.abs().max())
    return centered / denom if denom > 1e-12 else centered * 0.0


def _build_agentic_asset_views(success_dates: set[pd.Timestamp]) -> pd.DataFrame:
    returns_path = PROCESSED / "asset_returns_daily.parquet"
    if not returns_path.exists() or not success_dates:
        return pd.DataFrame()
    returns = pd.read_parquet(returns_path)
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce").dt.normalize()
    returns["asset_code"] = returns["asset_code"].astype(str)
    returns["return_1d"] = pd.to_numeric(returns["return_1d"], errors="coerce")
    returns = returns.dropna(subset=["date", "asset_code"]).sort_values(["date", "asset_code"])

    rows: list[dict[str, object]] = []
    available_set = set(returns["asset_code"].unique())
    for date in sorted(success_dates):
        history = returns[returns["date"].le(date)].copy()
        for module, codes in ASSET_VIEW_MODULES.items():
            available_codes = [code for code in codes if code in available_set]
            if len(available_codes) < 2:
                continue
            scores = _score_assets(history, available_codes)
            if len(scores) < 2:
                continue
            source = str(scores.idxmax())
            target = str(scores.idxmin())
            spread = float(scores.loc[source] - scores.loc[target])
            if spread < 0.70:
                continue
            confidence = round(float(np.clip(0.50 + 0.12 * min(spread, 1.5), 0.50, 0.68)), 3)
            rows.append(
                {
                    "date": date,
                    "source_asset": source,
                    "target_asset": target,
                    "relation": "overweight",
                    "confidence": confidence,
                    "raw_confidence": confidence,
                    "confidence_calibration": "asset_level_same_sleeve_momentum_rank_no_future_data",
                    "reason_tags": json.dumps(
                        [f"asset_level_{module}", "same_sleeve_relative_view", "uses_trailing_daily_returns_only"],
                        ensure_ascii=False,
                    ),
                    "view_type": "relative",
                    "generator": "agentic_asset_level_overlay_v1",
                    "validator_status": "valid",
                    "validator_reason": "",
                    "horizon_days": 20,
                    "model_alignment": "asset_level_agentic_extension",
                    "risk_flags": json.dumps([], ensure_ascii=False),
                    "invalidation_conditions": json.dumps(
                        ["trailing_rank_reverses", "module_state_blocks_risk_taking"],
                        ensure_ascii=False,
                    ),
                    "direction_strength": round(min(spread / 2.0, 1.0), 3),
                    "evidence_quality": 0.58,
                    "cross_module_confirmation": 0.50,
                    "contradiction_penalty": 0.12,
                    "liquidity_penalty": 0.08,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    mechanical_path = PROCESSED / "market_skill_views_mechanical_daily_v1.parquet"
    agentic_month_path = PROCESSED / "market_skill_views_agentic_daily_v1.parquet"
    audit_path = REPORTS / "agentic_market_view_audit.jsonl"

    mechanical = _normalize_dates(pd.read_parquet(mechanical_path))
    agentic_month = _normalize_dates(pd.read_parquet(agentic_month_path))
    success_dates = _read_success_dates(audit_path)

    agentic_rows = agentic_month[
        agentic_month["date"].isin(success_dates)
        & agentic_month.get("generator", "").astype(str).eq("agentic_market_view_v1")
    ].copy()
    if agentic_rows.empty and success_dates:
        raise RuntimeError("No non-fallback agentic rows found for successful audit dates.")
    asset_level_views = _build_agentic_asset_views(success_dates)

    # low_vol wants the stronger defensive overlay: replace whole successful months.
    low_raw = pd.concat([agentic_month.copy(), asset_level_views], ignore_index=True, sort=False)
    low_ep_rows = _write_ep(
        low_raw,
        PROCESSED / "market_skill_views_agentic_low_vol_daily_v1.parquet",
        PROCESSED / "ep_views_agentic_low_vol_daily_v1.parquet",
    )

    # balanced/high_vol should keep the mechanical daily context and only replace the exact
    # month-end decision card when the agent produced a valid non-fallback answer.
    blended_raw = pd.concat(
        [mechanical[~mechanical["date"].isin(success_dates)].copy(), agentic_rows, asset_level_views],
        ignore_index=True,
        sort=False,
    )
    blended_ep_rows = _write_ep(
        blended_raw,
        PROCESSED / "market_skill_views_agentic_blended_daily_v1.parquet",
        PROCESSED / "ep_views_agentic_balanced_daily_v1.parquet",
    )
    _write_ep(
        blended_raw,
        PROCESSED / "market_skill_views_agentic_blended_daily_v1.parquet",
        PROCESSED / "ep_views_agentic_high_vol_daily_v1.parquet",
    )

    report = {
        "success_agentic_dates": len(success_dates),
        "agentic_rows": len(agentic_rows),
        "agentic_asset_level_rows": len(asset_level_views),
        "mechanical_rows": len(mechanical),
        "low_vol_raw_rows": len(low_raw),
        "low_vol_ep_rows": low_ep_rows,
        "balanced_high_vol_raw_rows": len(blended_raw),
        "balanced_high_vol_ep_rows": blended_ep_rows,
        "low_vol_ep": "data_processed/ep_views_agentic_low_vol_daily_v1.parquet",
        "balanced_ep": "data_processed/ep_views_agentic_balanced_daily_v1.parquet",
        "high_vol_ep": "data_processed/ep_views_agentic_high_vol_daily_v1.parquet",
    }
    (REPORTS / "agentic_profile_view_sets_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
