from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CARDS = ROOT / "data_processed" / "market_state_cards_daily_v1.parquet"
VIEWS = ROOT / "data_processed" / "market_skill_views_mechanical_daily_v1.parquet"
EP_VIEWS = ROOT / "data_processed" / "ep_views_daily_v1.parquet"
AUDIT_CSV = ROOT / "output" / "reports" / "market_state_connection_audit.csv"
AUDIT_MD = ROOT / "output" / "reports" / "market_state_connection_audit.md"
REPAIRED_ALT_ASSETS = ["alt_nonferrous", "alt_chemical", "alt_us_bond"]


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def main() -> None:
    cards = pd.read_parquet(CARDS).sort_values("date").reset_index(drop=True)
    views = pd.read_parquet(VIEWS).sort_values("date").reset_index(drop=True)
    ep_views = pd.read_parquet(EP_VIEWS).sort_values("date").reset_index(drop=True)
    raw_alt = pd.read_parquet(
        ROOT / "data_processed" / "asset_returns_daily.parquet",
        columns=["date", "asset_code", "return_1d"],
    )
    raw_alt = raw_alt[raw_alt["asset_code"].isin(REPAIRED_ALT_ASSETS)].copy()
    raw_alt["date"] = pd.to_datetime(raw_alt["date"], errors="coerce")
    raw_returns = {
        (pd.Timestamp(row.date), str(row.asset_code)): row.return_1d
        for row in raw_alt.drop_duplicates(["date", "asset_code"], keep="last").itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []

    def add(check: str, violations: int, samples: int, details: str) -> None:
        rows.append(
            {
                "check": check,
                "status": "PASS" if violations == 0 else "FAIL",
                "violations": int(violations),
                "samples": int(samples),
                "details": details,
            }
        )

    ret1_differences: list[float] = []
    ret20_differences: list[float] = []
    missing_ret20 = 0
    exact_ret1_samples = 0
    ret20_samples = 0
    context_missing = 0
    context_samples = 0
    holding_without_data = 0
    overweight_without_holding = 0

    for card in cards.itertuples(index=False):
        alt = _loads(card.alt_detail_states).get("assets", {})
        assets = _loads(card.asset_detail_states).get("assets", {})
        for asset_code, item in assets.items():
            context = item.get("domain_context", {})
            context_samples += 1
            required = {"data_available", "holding_eligible", "overweight_eligible"}
            if not required.issubset(context):
                context_missing += 1
                continue
            if context["holding_eligible"] and not context["data_available"]:
                holding_without_data += 1
            if context["overweight_eligible"] and not context["holding_eligible"]:
                overweight_without_holding += 1

        for asset_code in REPAIRED_ALT_ASSETS:
            detail = alt.get(asset_code, {})
            asset = assets.get(asset_code, {})
            metrics = detail.get("metrics", {})
            asset_metrics = asset.get("metrics", {})
            available = bool(asset.get("data_quality", {}).get("available", False))
            if available and not _finite(metrics.get("ret_20d")):
                missing_ret20 += 1
            if _finite(metrics.get("ret_20d")) and _finite(asset_metrics.get("ret_20d")):
                ret20_samples += 1
                ret20_differences.append(
                    abs(float(metrics["ret_20d"]) - float(asset_metrics["ret_20d"]))
                )
            age = asset_metrics.get("observation_age_days")
            if _finite(age) and float(age) == 0 and _finite(asset_metrics.get("ret_5d")):
                if _finite(metrics.get("ret_1d")):
                    exact_ret1_samples += 1
                    # The detail layer rounds daily returns to five decimals.
                    raw_return = raw_returns.get((pd.Timestamp(card.date), asset_code))
                    if _finite(raw_return):
                        ret1_differences.append(
                            abs(float(metrics["ret_1d"]) - float(raw_return))
                        )

    add(
        "asset_context_has_explicit_eligibility_fields",
        context_missing,
        context_samples,
        "Every asset card must separate data availability, holding eligibility and overweight eligibility.",
    )
    add(
        "holding_requires_available_data",
        holding_without_data,
        context_samples,
        "holding_eligible cannot be true when data_available is false.",
    )
    add(
        "overweight_requires_holding_eligibility",
        overweight_without_holding,
        context_samples,
        "overweight_eligible implies holding_eligible.",
    )
    add(
        "repaired_alt_ret20_populated_when_asset_available",
        missing_ret20,
        ret20_samples + missing_ret20,
        "Nonferrous, chemical and US-bond 20-day returns must reach ALT detail states.",
    )
    add(
        "repaired_alt_ret20_matches_asset_card",
        int(sum(value > 5e-5 for value in ret20_differences)),
        ret20_samples,
        f"max_abs_difference={max(ret20_differences, default=np.nan):.8f}",
    )
    add(
        "repaired_alt_ret1_uses_exact_observation_date",
        int(sum(value > 1.1e-5 for value in ret1_differences)),
        exact_ret1_samples,
        f"max_abs_difference={max(ret1_differences, default=np.nan):.8f}; no as-of carry for returns",
    )

    latest = cards.iloc[-1]
    latest_assets = _loads(latest["asset_detail_states"]).get("assets", {})
    latest_alt = _loads(latest["alt_detail_states"])
    aaa = latest_assets.get("credit_ind_aaa", {}).get("domain_context", {})
    add(
        "latest_aaa_credit_holding_vs_overweight_semantics",
        int(not (aaa.get("holding_eligible") is True and aaa.get("overweight_eligible") is False)),
        1,
        f"holding_eligible={aaa.get('holding_eligible')}; overweight_eligible={aaa.get('overweight_eligible')}",
    )

    latest_date = pd.Timestamp(latest["date"])
    latest_views = views[pd.to_datetime(views["date"]).eq(latest_date)]
    expected_alt4_view = bool(
        (latest_views["source_asset"].eq("rates_long_duration")
        & latest_views["target_asset"].eq("alternatives")
        & latest_views["validator_status"].eq("valid")).any()
    ) if len(latest_views) else False
    add(
        "latest_alt4_state_reaches_mechanical_views",
        int(latest_alt.get("alt_state") == "ALT4" and not expected_alt4_view),
        1,
        f"alt_state={latest_alt.get('alt_state')}; rates_over_alternatives_view={expected_alt4_view}",
    )

    valid_views = views[views["validator_status"].eq("valid")]
    add(
        "ep_views_match_valid_mechanical_views",
        abs(len(valid_views) - len(ep_views)),
        len(valid_views),
        f"valid_mechanical={len(valid_views)}; ep_views={len(ep_views)}",
    )

    audit = pd.DataFrame(rows)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    lines = [
        "# Market State Connection Audit",
        "",
        f"- card range: `{pd.Timestamp(cards['date'].min()).date()}` to `{pd.Timestamp(cards['date'].max()).date()}`",
        f"- card rows: {len(cards)}",
        f"- checks passed: {int(audit['status'].eq('PASS').sum())}/{len(audit)}",
        "",
        audit.to_markdown(index=False),
    ]
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(audit.to_string(index=False))
    if audit["status"].ne("PASS").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
