"""Identify box conflicts without tuning bounds or reading future performance."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v1_strategy import load_config, RISK_SLEEVES, sleeve_bounds
from src.v75_strict_anchor import covariance_complete
from src.v78_exact_risk_budget import exact_positive_budget

OUT = ROOT / "output/reports/risk_budget_compatibility_v78"
DEST = ROOT / "data_processed/risk_budget_compatibility_v78"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    sources = {ROOT / "config/v1_strategy_config.yaml", ROOT / "src/v75_strict_anchor.py",
               ROOT / "src/v78_exact_risk_budget.py", Path(__file__)}
    rows, conflicts, weights = [], [], []
    for profile in ("low_vol", "balanced", "high_vol"):
        spec = copy.deepcopy(cfg)
        spec["product"]["selected_profile"] = profile
        paths = [ROOT / f"data_processed/strict_anchors_v75/{profile}_sleeve_returns_monthly.parquet",
                 ROOT / f"data_processed/strict_anchors_v75/{profile}_risk_budget_base_monthly.parquet"]
        sources.update(paths)
        history, anchors = [pd.read_parquet(p) for p in paths]
        budget = np.array([spec["product"]["profiles"][profile]["target_risk_contribution"][s] for s in RISK_SLEEVES])
        budget /= budget.sum()
        bounds = sleeve_bounds(spec)
        for row in anchors.itertuples(index=False):
            c, n = covariance_complete(history.loc[:row.date].tail(36),
                float(spec["risk_budget"]["covariance_shrinkage_to_diag"]), float(spec["risk_budget"]["diagonal_vol_floor"]))
            for method, b in (("exact_product_risk_budget_no_product_bounds", budget),
                              ("classic_ERC_no_product_bounds", np.ones(5) / 5)):
                w, audit = exact_positive_budget(c, b)
                lo, hi = [np.array([bounds[s][i] for s in RISK_SLEEVES]) for i in (0, 1)]
                violation = np.maximum(np.maximum(lo - w, w - hi), 0)
                rows.append(dict(profile=profile, signal_date=row.signal_date, method=method, complete_months=n,
                    solver=audit["solver"], primary_status=audit["primary_status"],
                    max_rc_error=audit["max_rc_error"], max_box_violation=float(violation.max()),
                    compatible_with_product_box=bool(violation.max() <= 1e-6),
                    annual_forecast_vol=float(np.sqrt(w @ c @ w * 12))))
                weights.append(dict(profile=profile, date=row.date, signal_date=row.signal_date,
                    effective_date=row.effective_date, method=method, **dict(zip(RISK_SLEEVES, map(float, w))), cash=0.))
                for i, sleeve in enumerate(RISK_SLEEVES):
                    if violation[i] > 1e-6:
                        conflicts.append(dict(profile=profile, signal_date=row.signal_date, method=method,
                            sleeve=sleeve, exact_weight=float(w[i]), current_lower=float(lo[i]), current_upper=float(hi[i]),
                            violation=float(violation[i]), required_direction="below_lower" if w[i] < lo[i] else "above_upper",
                            bounds_modified=False))
        print(f"V78 {profile}: exact-budget and classic ERC diagnostics complete", flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "exact_solution_compatibility.csv", index=False)
    pd.DataFrame(conflicts).to_csv(OUT / "conflicting_bounds.csv", index=False)
    pd.DataFrame(weights).to_parquet(DEST / "classic_erc_and_exact_budget_weights.parquet", index=False)
    summary = detail.groupby(["profile", "method"]).agg(months=("compatible_with_product_box", "size"),
        compatible_months=("compatible_with_product_box", "sum"), maximum_rc_residual=("max_rc_error", "max"),
        mean_box_violation=("max_box_violation", "mean"), maximum_box_violation=("max_box_violation", "max")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    status = dict(status="exact_budget_compatibility_audited_no_parameter_change", solutions=len(detail),
        maximum_rc_residual=float(detail.max_rc_error.max()), product_bounds_modified=False,
        no_test_pnl_used=True, baseline_is_unconstrained_reference_not_product_replacement=True,
        portfolio_backtest_run=False, models_retrained=False, formal_promotion=False,
        mathematical_scope="SPD and positive budget imply a unique normalized RB solution; box conflict diagnosed within numerical tolerance, not interval arithmetic proof")
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (OUT / "source_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
