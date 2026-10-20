"""Freeze product box-first research contract without changing existing parameters."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v1_strategy import load_config, RISK_SLEEVES


def main():
    out = ROOT / "output/reports/product_contract_v80"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    contract = dict(version="v80_box_first_research_contract", status="research_only_not_promoted",
        priorities=["source_and_visibility", "long_only_no_borrowing", "target_product_bounds",
                    "execution_and_cash_availability", "risk_budget_soft_fit"],
        target_sum=1., target_cash_weight=0., cost_per_gross_traded_notional=.0005,
        signal_execution="strictly_next_domestic_session_close_not_same_day",
        risk_budget_exact_required=False, actual_rc_and_error_disclosure_required=True,
        realized_risk_ordering_guaranteed=False,
        blocked_execution_policy="keep_inventory_flag_breach_do_not_fake_compliant_fill",
        settlement_cash_policy="not_an_allocatable_risk_sleeve_no_automatic_borrowing_against_receivables",
        baseline_classic_erc="separate_unbounded_reference_not_a_box_compliant_product",
        assets=38, nine_grid_preserved=True, models=["iTransformer_relative_views", "PPO_adaptive_execution"],
        training_allowed=False, training_blockers=["cashflow_and_inventory_certification", "same_basis_mechanical_baselines"],
        formal_promotion=False, profiles=copy.deepcopy(cfg["product"]["profiles"]))
    config_path = ROOT / "config/v80_product_contract_research.yaml"
    config_path.write_text(yaml.safe_dump(contract, allow_unicode=False, sort_keys=False), encoding="utf-8")
    rows = []
    for profile in ("low_vol", "balanced", "high_vol"):
        path = ROOT / f"data_processed/strict_anchors_v75/{profile}_risk_budget_base_monthly.parquet"
        weights = pd.read_parquet(path)
        bounds = cfg["product"]["profiles"][profile]["sleeve_bounds"]
        w = weights[RISK_SLEEVES].to_numpy()
        lo, hi = [np.array([bounds[s][i] for s in RISK_SLEEVES]) for i in (0, 1)]
        for i, row in enumerate(weights.itertuples(index=False)):
            residual = float(max(0, np.max(lo - w[i]), np.max(w[i] - hi)))
            passed = (abs(w[i].sum() - 1) <= 1e-8 and residual <= 1e-8
                      and row.cash == 0 and pd.Timestamp(row.effective_date) > pd.Timestamp(row.signal_date))
            if not passed:
                raise ValueError(f"Anchor breaks hard product contract: {profile} {row.date}")
            rows.append(dict(profile=profile, signal_date=row.signal_date, budget_residual=float(abs(w[i].sum() - 1)),
                             box_residual=residual, same_day_execution=False, cash_target=0., hard_contract_passed=True))
    pd.DataFrame(rows).to_csv(out / "existing_strict_anchor_contract_checks.csv", index=False)
    status = dict(status="research_product_contract_frozen", anchors_checked=len(rows),
        hard_contract_passed=True, product_parameters_changed=False, risk_budget_priority="soft",
        old_config_overwritten=False, formal_training_allowed=False, ppo_started=False, formal_promotion=False)
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    inputs = [ROOT / "config/v1_strategy_config.yaml", config_path, Path(__file__)]
    inputs += [ROOT / f"data_processed/strict_anchors_v75/{p}_risk_budget_base_monthly.parquet"
               for p in ("low_vol", "balanced", "high_vol")]
    (out / "source_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in inputs}, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
