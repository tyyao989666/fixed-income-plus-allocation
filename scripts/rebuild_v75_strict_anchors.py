"""Isolated three-product anchors from V74, no old cards, fills or forecasts."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v1_strategy import (load_config, RISK_SLEEVES, _preferred_asset_weight_map,
                             solve_risk_budget, sleeve_bounds)
from src.v75_strict_anchor import frozen_monthly_sleeves, covariance_complete, solve_budget, risk_shares

OUT = ROOT / "output/reports/strict_anchors_v75"
DEST = ROOT / "data_processed/strict_anchors_v75"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "data_processed/correct_credit_v74/research_returns_daily_38.parquet",
              ROOT / "data_processed/correct_credit_v74/dataset.npz",
              ROOT / "output/reports/correct_credit_v74/status.json",
              ROOT / "config/v1_strategy_config.yaml", ROOT / "src/v1_strategy.py",
              ROOT / "src/v75_strict_anchor.py", Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    status = json.loads(inputs[2].read_text(encoding="utf-8"))
    assert sha(inputs[1]) == status["dataset_sha256"]
    daily = pd.read_parquet(inputs[0])
    cutoff = daily.index[-1] - pd.offsets.MonthEnd(1)
    cfg = load_config()
    audit, all_weights, attempts, gaps, old_comparisons, memberships, policies = [], [], [], [], [], [], []
    for profile in ("low_vol", "balanced", "high_vol"):
        spec = copy.deepcopy(cfg)
        spec["product"]["selected_profile"] = profile
        preferences = {s: _preferred_asset_weight_map(s, spec) for s in RISK_SLEEVES}
        long, members = frozen_monthly_sleeves(daily, preferences, cutoff)
        history = long.pivot(index="date", columns="sleeve", values="return_1m").reindex(columns=RISK_SLEEVES)
        history.to_parquet(DEST / f"{profile}_sleeve_returns_monthly.parquet")
        members["profile"] = profile
        memberships.extend(members.to_dict("records"))
        gaps.extend(long.loc[long.return_1m.isna()].assign(profile=profile).to_dict("records"))
        bounds = sleeve_bounds(spec)
        lo, hi = [np.array([bounds[s][i] for s in RISK_SLEEVES]) for i in (0, 1)]
        target = np.array([spec["product"]["profiles"][profile]["target_risk_contribution"][s] for s in RISK_SLEEVES])
        target /= target.sum()
        policies.append(dict(profile=profile, preferred_asset_weights=preferences, target_rc=target.tolist(),
            lower=lo.tolist(), upper=hi.tolist(), source_config_unchanged=True))
        profile_rows = []
        for date in history.index:
            window = history.loc[:date].tail(int(spec["risk_budget"]["lookback_months"]))
            n = int(window.notna().all(axis=1).sum())
            if n < int(spec["risk_budget"]["min_history_months"]):
                audit.append(dict(profile=profile, signal_date=date, status="warmup_no_anchor",
                                  complete_months=n, weight_sum=np.nan))
                continue
            c, n = covariance_complete(window, float(spec["risk_budget"]["covariance_shrinkage_to_diag"]),
                                        float(spec["risk_budget"]["diagonal_vol_floor"]))
            signal = daily.index[daily.index <= date][-1]
            future = daily.index[daily.index > signal]
            effective = future[0] if len(future) else pd.NaT
            for method, budget in (("product_risk_budget", target), ("equal_risk_same_bounds", np.ones(5) / 5)):
                w, diag = solve_budget(c, budget, lo, hi, int(spec["risk_budget"]["max_iterations"]))
                row = dict(date=date, signal_date=signal, effective_date=effective, method=method, profile=profile,
                    **dict(zip(RISK_SLEEVES, map(float, w))), cash=0.)
                all_weights.append(row)
                if method == "product_risk_budget":
                    profile_rows.append(row)
                audit.append(dict(profile=profile, signal_date=signal, method=method, status="converged_feasible",
                    complete_months=n, window_calendar_months=len(window), weight_sum=float(w.sum()),
                    minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(c).min()),
                    max_rc_error=diag["max_rc_error"], projected_gradient_residual=diag["projected_gradient_residual"],
                    exact_target=diag["risk_target_exact"], forecast_annual_vol=float(np.sqrt(w @ c @ w * 12)),
                    global_optimum_proven=False))
                attempts.extend(dict(profile=profile, signal_date=str(signal.date()), method=method,
                                     start_number=i, **v) for i, v in enumerate(diag["attempts"]))
                if method == "product_risk_budget":
                    legacy = solve_risk_budget(c, target, bounds, spec)
                    old_comparisons.append(dict(profile=profile, signal_date=signal,
                        strict_squared_rc_error=diag["objective"],
                        legacy_squared_rc_error=float(np.sum((risk_shares(legacy, c) - target)**2)),
                        legacy_max_rc_error=float(np.max(np.abs(risk_shares(legacy, c) - target))),
                        legacy_bounds_residual=float(max(0, np.max(lo - legacy), np.max(legacy - hi)))))
        pd.DataFrame(profile_rows).to_parquet(DEST / f"{profile}_risk_budget_base_monthly.parquet", index=False)
        print(f"V75 {profile}: {len(profile_rows)} strict monthly product anchors", flush=True)
    weights = pd.DataFrame(all_weights)
    weights.to_parquet(DEST / "all_methods_three_profiles.parquet", index=False)
    pd.DataFrame(audit).to_csv(OUT / "solver_and_covariance_audit.csv", index=False)
    pd.DataFrame(attempts).to_json(OUT / "solver_attempts.json", orient="records", indent=2)
    pd.DataFrame(old_comparisons).to_csv(OUT / "legacy_solver_comparison_same_inputs.csv", index=False)
    pd.DataFrame(gaps).to_csv(OUT / "masked_sleeve_months.csv", index=False)
    pd.DataFrame(memberships).to_parquet(DEST / "fixed_signal_monthly_membership.parquet", index=False)
    summary = weights.loc[weights.method.eq("product_risk_budget")].groupby("profile")[RISK_SLEEVES].mean()
    summary.to_csv(OUT / "average_sleeve_weights.csv")
    result = dict(status="strict_anchors_built_research_only", cutoff=str(cutoff.date()), assets=38,
        methods=["product_risk_budget", "equal_risk_same_bounds"], source="V74 corrected returns only",
        missing_sleeve_months_not_zero_filled=len(gaps), old_cards_or_predictions_used=False,
        old_anchor_read=False, warmup_policy="no_anchor_until_12_complete_months_within_36_calendar_months",
        covariance_policy="complete_five_sleeve_months_20pct_diagonal_shrinkage_original_floor",
        solver_global_optimality_claimed=False, historical_returns_execution_certified=False,
        formal_training_allowed=False, ppo_started=False, formal_promotion=False,
        semantic_note="Within-sleeve references are fixed-at-prior-close synthetic buy-and-hold paths, not inventory NAV")
    (OUT / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "policy_snapshot.json").write_text(json.dumps(policies, indent=2), encoding="utf-8")
    assert all(sha(ROOT / k) == v for k, v in hashes.items())
    (OUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
