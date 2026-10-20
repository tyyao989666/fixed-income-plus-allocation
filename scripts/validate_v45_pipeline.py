"""Independent checks of saved v45 artifacts; not a source-PIT certification."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v45_data import annual_splits
from src.v45_portfolio import frame_hash, config, RISK_SLEEVES, _strict_monthly_anchor


def main():
    reports = ROOT / "output/reports"
    checks = []

    def check(name, condition, detail=None):
        checks.append(dict(check=name, passed=bool(condition), detail=detail))

    metadata = json.loads((reports / "v45_metadata.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "data_processed/v45_dataset.npz", allow_pickle=False) as raw:
        a = {key: raw[key] for key in raw.files}
    calendar = pd.DatetimeIndex(pd.to_datetime(a["dates"]))
    with np.load(ROOT / "data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz", allow_pickle=False) as frozen:
        check("frozen_return_values_unchanged", np.array_equal(a["returns"], frozen["returns"], equal_nan=True))
        check("frozen_observation_masks_unchanged", np.array_equal(a["return_masks"], frozen["return_masks"]))
    check("38_context_assets", len(metadata["assets"]) == 38 and a["local_values"].shape[1] == 38)
    check("frozen_research_dates", calendar[0] == pd.Timestamp("2018-01-02") and calendar[-1] == pd.Timestamp("2026-07-15"))
    check("observed_features_finite", np.isfinite(a["local_values"][a["local_masks"].astype(bool)]).all())
    check("nine_grid_and_two_scores", a["macro_values"].shape[1] == 11)
    for year in range(2022, 2027):
        split = annual_splits(a, year)
        cutoff = pd.Timestamp(year, 1, 1).value
        valid = not split.get("reason")
        for key in ("train", "validation", "gate"):
            indices = split[key]
            valid &= bool(len(indices) and (a["dates"][indices] < cutoff).all()
                and (a["label_end"][indices] < cutoff).all())
        check(f"{year}_T_V_G_labels_preyear", valid)

    forecast = json.loads((reports / "v45_forecast_report.json").read_text(encoding="utf-8"))
    check("forecast_full_not_debug", forecast.get("debug") is False)
    folds = forecast.get("folds", [])
    check("five_forecast_years", [f["year"] for f in folds] == list(range(2022, 2027)))
    for fold in folds:
        stages = fold.get("stages", [])
        year = fold["year"]
        check(f"{year}_nine_real_neural_fits", len(stages) == 9)
        for index, stage in enumerate(stages):
            pretraining = stage["stage"] == "pretraining"
            lower = 100 if pretraining else 50
            check(f"{year}_{index}_training_budget", lower <= stage["trained_epochs"] <= 1000,
                dict(stage=stage["stage"], trained=stage["trained_epochs"], selected=stage["selected_epoch"]))
            selected = stage["selected_epoch"]
            explicit_shadow = (selected == 0 and stage.get("selection_available") is False
                and stage.get("status") == "shadow_validation_score_unavailable")
            check(f"{year}_{index}_checkpoint_is_trained_or_explicit_shadow",
                1 <= selected <= stage["trained_epochs"] or explicit_shadow)
        check(f"{year}_all_encoders_actually_changed", all(s.get("encoder_parameters_changed", True) for s in stages))

    predictions = pd.read_parquet(ROOT / "data_processed/v45_predictions.parquet")
    check("all_four_forecast_candidates", set(predictions.model) == {"F0", "F1", "F2", "F3"})
    check("prediction_keys_unique", not predictions.duplicated(["as_of_date", "task", "entity", "horizon_days", "model", "seed"]).any())
    check("all_active_increments_finite", np.isfinite(predictions.active_increment).all())
    live = predictions.active_increment.ne(0)
    active = predictions.loc[live]
    check("only_selected_ensemble_active", (active.seed.eq(-1) & active.model.eq(active.selected_candidate)
        & active.component_gate_passed & active.active_component).all())
    check("active_forecast_causal_times", (pd.to_datetime(active.train_end_date) < pd.to_datetime(active.as_of_date)).all()
        and (pd.to_datetime(active.gate_as_of_date) < pd.to_datetime(active.as_of_date)).all())
    check("inactive_components_zero_increment", predictions.loc[~predictions.active_component, "active_increment"].eq(0).all())
    for fold in folds:
        if not fold.get("selection_available", False):
            rows = pd.to_datetime(predictions.as_of_date).dt.year.eq(fold["year"])
            check(f"{fold['year']}_unselectable_fold_stays_shadow", not predictions.loc[rows, "active_component"].any())

    expected_dates = calendar[(calendar >= "2022-01-05") & (calendar <= "2026-07-15")]
    stage = json.loads((reports / "v45_portfolio_stage_gate.json").read_text(encoding="utf-8"))
    cfg = config()
    check("portfolio_training_requested", stage.get("training_requested") is True)
    check("portfolio_engineering_runner_pass", stage.get("engineering_passed") is True)
    for profile in ("low_vol", "balanced", "high_vol"):
        manifest = json.loads((reports / f"v45_{profile}_target_manifest.json").read_text(encoding="utf-8"))
        a2 = pd.read_parquet(ROOT / f"data_processed/v45_{profile}_A2_targets.parquet")
        a3 = pd.read_parquet(ROOT / f"data_processed/v45_{profile}_A3_targets.parquet")
        check(f"{profile}_same_A2_A3_target", frame_hash(a2) == frame_hash(a3) == manifest["target_hashes"]["A2"])
        prior, modules = _strict_monthly_anchor(profile, calendar, cfg)
        for layer in ("A0", "A1", "A2", "A3", "R0", "R1"):
            prefix = f"{profile}_{layer}"
            target = pd.read_parquet(ROOT / f"data_processed/v45_{prefix}_targets.parquet")
            diagnostic = pd.read_parquet(ROOT / f"data_processed/v45_{prefix}_diagnostics.parquet").reindex(target.index)
            sleeves = pd.DataFrame({s: target[[asset for asset in target if modules[asset] == s]].sum(axis=1) for s in RISK_SLEEVES})
            anchor = diagnostic[[f"anchor_{s}" for s in RISK_SLEEVES]].set_axis(RISK_SLEEVES, axis=1)
            lower = diagnostic[[f"lower_{s}" for s in RISK_SLEEVES]].to_numpy()
            upper = diagnostic[[f"upper_{s}" for s in RISK_SLEEVES]].to_numpy()
            check(f"{prefix}_target_budget", np.allclose(target.sum(axis=1), 1, atol=1e-8, rtol=0) and target.min().min() >= -1e-10)
            check(f"{prefix}_target_box", ((sleeves.to_numpy() >= lower-1e-6) & (sleeves.to_numpy() <= upper+1e-6)).all())
            check(f"{prefix}_target_active_l1", ((sleeves-anchor).abs().sum(axis=1) <= cfg["portfolio"]["daily_active_limits"][profile]+1e-6).all())
            check(f"{prefix}_target_change_l1", (sleeves.diff().abs().sum(axis=1) <= cfg["portfolio"]["daily_target_change_limits"][profile]+1e-6).all())
            for sleeve in RISK_SLEEVES:
                assets = [asset for asset in target if modules[asset] == sleeve]
                shares = prior.loc[target.index, assets].div(prior.loc[target.index, assets].sum(axis=1).replace(0, np.nan), axis=0)
                reference = shares.fillna(0).mul(sleeves[sleeve], axis=0)
                check(f"{prefix}_{sleeve}_asset_concentration", (target[assets].le(.75*sleeves[sleeve]+1e-8, axis=0)).all().all())
                check(f"{prefix}_{sleeve}_asset_active_limit", ((target[assets]-reference).abs() <= cfg["portfolio"]["asset_active_limits"][profile]+1e-8).all().all())
                check(f"{prefix}_{sleeve}_no_new_ineligible_asset", not (target[assets].gt(1e-8) & shares.fillna(0).le(1e-12)).any().any())
            daily = pd.read_csv(reports / f"v45_{prefix}_daily.csv", parse_dates=["date", "signal_date"])
            holdings = pd.read_parquet(ROOT / f"data_processed/v45_{prefix}_holdings.parquet")
            weights = holdings.pivot(index="date", columns="asset_code", values="weight").sort_index()
            end = holdings.pivot(index="date", columns="asset_code", values="end_weight").reindex_like(weights)
            check(f"{prefix}_same_calendar", pd.DatetimeIndex(daily.date).equals(expected_dates))
            finite_cols = ["strategy_return", "gross_return", "nav", "turnover", "transaction_cost"]
            check(f"{prefix}_finite", np.isfinite(daily[finite_cols]).all().all() and np.isfinite(weights).all().all())
            signal_positions = calendar.get_indexer(daily.signal_date)
            execution_positions = calendar.get_indexer(daily.date)
            check(f"{prefix}_T_plus_1", ((execution_positions - signal_positions) == 1).all() and (signal_positions >= 0).all())
            check(f"{prefix}_net_gross_cost", np.allclose(daily.strategy_return, daily.gross_return-daily.transaction_cost, atol=1e-12, rtol=0))
            check(f"{prefix}_actual_cost_5bp", np.allclose(daily.transaction_cost, daily.turnover*.0005, atol=1e-12, rtol=0))
            check(f"{prefix}_nav_recurrence", np.allclose(daily.nav, np.cumprod(1+daily.strategy_return), atol=1e-10, rtol=1e-10))
            check(f"{prefix}_budget_no_short", np.allclose(weights.sum(axis=1), 1, atol=1e-8, rtol=0) and weights.min().min() >= -1e-8)
            check(f"{prefix}_end_budget", np.allclose(end.sum(axis=1), 1, atol=1e-8, rtol=0))
            check(f"{prefix}_no_cash", not any("cash" in str(asset).lower() for asset in weights))
            actual = (weights.iloc[1:].to_numpy()-end.iloc[:-1].to_numpy())
            check(f"{prefix}_turnover_from_drifted_book", np.allclose(np.abs(actual).sum(axis=1), daily.turnover.iloc[1:], atol=1e-8, rtol=0))
            check(f"{prefix}_gross_from_asset_contributions", np.allclose(
                holdings.groupby("date").return_contribution.sum().reindex(daily.date).to_numpy(), daily.gross_return, atol=1e-10, rtol=0))
        profile_folds = stage.get("folds", {}).get(profile, [])
        check(f"{profile}_three_PPO_years", [f["year"] for f in profile_folds] == [2024, 2025, 2026])
        for fold in profile_folds:
            check(f"{profile}_{fold['year']}_four_PPO_candidates", len(fold.get("candidate_selection_edges", [])) == 4)
            split = fold.get("split", {})
            cutoff = pd.Timestamp(fold["year"], 1, 1)
            check(f"{profile}_{fold['year']}_PPO_segments_causal", len(split) == 3 and all(
                pd.Timestamp(part["last_execution"]) < cutoff for part in split.values()))
        daily = pd.read_csv(reports / f"v45_{profile}_A3_daily.csv")
        check(f"{profile}_PPO_never_deployed_without_forecast", not (daily.model_used & daily.active_components.eq(0)).any())
    result = dict(engineering_passed=all(c["passed"] for c in checks), checks=checks,
        failed=[c for c in checks if not c["passed"]], active_forecast_rows=int(live.sum()),
        source_PIT_certified=False, untouched_holdout=False, formal_promotion=False,
        caveat="Engineering chronology checks do not certify proxy total returns, historical memberships, source publication vintages, or future alpha.")
    (reports / "v45_final_integrity_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(reports / "v45_final_integrity_checks.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: v for k,v in result.items() if k != "checks"}, ensure_ascii=False, indent=2))
    if not result["engineering_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
