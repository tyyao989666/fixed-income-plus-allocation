"""Remove legacy V44 target-anchor dependency; keep execution/training gates closed."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v74_rebuild import rebuild
from src.v75_strict_anchor import frozen_monthly_sleeves, covariance_complete, solve_budget
from src.v1_strategy import load_config, RISK_SLEEVES, _preferred_asset_weight_map, sleeve_bounds
from src.v45_data import annual_splits

OUT = ROOT / "output/reports/clean_reference_v77"
DEST = ROOT / "data_processed/clean_reference_v77"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    inputs = {Path(__file__), ROOT / "src/v74_rebuild.py", ROOT / "src/v75_strict_anchor.py",
              ROOT / "src/v45_data.py", ROOT / "src/v53_unified_data.py", ROOT / "src/v55_learning_objective.py",
              ROOT / "src/v1_strategy.py", ROOT / "config/v1_strategy_config.yaml"}

    def read(relative):
        path = ROOT / relative
        inputs.add(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text(encoding="utf-8"))

    def save(name, value):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    # Require unmodified lineage, not just an 'OK' status string.
    for relative in ("output/reports/correct_credit_v74/source_hashes.json", "output/reports/strict_anchors_v75/source_hashes.json"):
        for name, expected in read(relative).items():
            path = ROOT / name
            inputs.add(path)
            assert sha(path) == expected, name
    daily = read("data_processed/correct_credit_v74/research_returns_daily_38.parquet")
    metadata = read("output/reports/correct_credit_v74/metadata.json")
    status = read("output/reports/correct_credit_v74/status.json")
    source = ROOT / "data_processed/correct_credit_v74/dataset.npz"
    inputs.add(source)
    assert sha(source) == status["dataset_sha256"]
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    anchor = read("data_processed/strict_anchors_v75/balanced_risk_budget_base_monthly.parquet")
    assert not anchor.signal_date.duplicated().any()
    aligned = pd.merge_asof(pd.DataFrame(dict(date=daily.index)),
        anchor[["signal_date", *RISK_SLEEVES]].sort_values("signal_date"),
        left_on="date", right_on="signal_date", direction="backward")
    bw = aligned[RISK_SLEEVES].to_numpy(dtype=float)
    known = np.isfinite(bw).all(axis=1)
    assert np.allclose(bw[known].sum(axis=1), 1)
    assert (aligned.loc[known, "signal_date"] <= aligned.loc[known, "date"]).all()
    bw[~known] = 0  # Unavailable reference, never a cash investment or observed return.
    arrays["balanced_reference_weights"] = bw
    prefix_checks = []
    cfg = load_config()
    for profile in ("low_vol", "balanced", "high_vol"):
        spec = copy.deepcopy(cfg)
        spec["product"]["selected_profile"] = profile
        pref = {s: _preferred_asset_weight_map(s, spec) for s in RISK_SLEEVES}
        targets = np.array([spec["product"]["profiles"][profile]["target_risk_contribution"][s] for s in RISK_SLEEVES])
        targets /= targets.sum()
        bounds = sleeve_bounds(spec)
        lo, hi = [np.array([bounds[s][i] for s in RISK_SLEEVES]) for i in (0, 1)]
        saved = read(f"data_processed/strict_anchors_v75/{profile}_risk_budget_base_monthly.parquet")
        # Recreate fixed-year checkpoints with ALL later daily observations removed.
        probes = saved.groupby(saved.signal_date.dt.year).tail(1).iloc[::2]
        for row in probes.itertuples(index=False):
            long, _ = frozen_monthly_sleeves(daily.loc[:row.signal_date], pref, row.date)
            hist = long.pivot(index="date", columns="sleeve", values="return_1m").reindex(columns=RISK_SLEEVES)
            c, n = covariance_complete(hist.loc[:row.date].tail(int(spec["risk_budget"]["lookback_months"])),
                float(spec["risk_budget"]["covariance_shrinkage_to_diag"]), float(spec["risk_budget"]["diagonal_vol_floor"]))
            w, _ = solve_budget(c, targets, lo, hi, int(spec["risk_budget"]["max_iterations"]))
            expected = np.array([getattr(row, s) for s in RISK_SLEEVES])
            assert np.allclose(w, expected, atol=1e-8, rtol=0)
            prefix_checks.append(dict(profile=profile, signal_date=row.signal_date,
                                      later_observations_removed=True, max_weight_difference=float(np.max(np.abs(w - expected)))))
    pd.DataFrame(prefix_checks).to_csv(OUT / "future_truncation_anchor_audit.csv", index=False)
    bars = {k: read(f"data_processed/unified_v53/source_{k}.parquet") for k in ("open", "high", "low", "close", "vol", "amount")}
    cb_ref = read("data_processed/unified_v53/cb_reference_returns.parquet").iloc[:, 0].to_numpy()
    print("V77 clean-reference rebuilding labels; no model training", flush=True)
    result, names = rebuild(arrays, daily, bars, metadata, cb_ref)
    assert np.array_equal(result["regime_index"], arrays["regime_index"])
    assert np.array_equal(result["return_masks"], arrays["return_masks"])
    assert np.array_equal(result["returns"], arrays["returns"])
    for j, h in enumerate((5, 20, 60)):
        assert not result["sleeve_target_mask"][-h:, j].any()
        assert not result["sleeve_raw_relative_target_mask"][~known, j].any()
    np.savez_compressed(DEST / "dataset.npz", **result)
    daily.to_parquet(DEST / "research_returns_daily_38.parquet")
    aligned.assign(reference_available=known).to_parquet(DEST / "daily_clean_reference_alignment.parquet", index=False)
    audits = []
    for year in range(2022, 2027):
        split = annual_splits(result, year)
        audits.append(dict(year=year, reason=split["reason"], **{k: len(split[k]) for k in ("train", "validation", "gate", "test")}))
    pd.DataFrame(audits).to_csv(OUT / "annual_split_readiness.csv", index=False)
    meta = copy.deepcopy(metadata)
    meta.update(version="v77_correct_credit_clean_risk_reference_research_only", dataset_sha256=sha(DEST / "dataset.npz"),
        parent_dataset_sha256=sha(source), balanced_reference_anchor_lineage_certified=True,
        reference_anchor_policy="V75 strict balanced product risk budget, available after signal close, no warmup fallback",
        local_features=names, old_checkpoints_incompatible=True,
        source_manifest={str(p.relative_to(ROOT)): dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sorted(inputs)},
        array_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in result.items()})
    # Replace stale warning rather than claiming the remaining return contracts are solved.
    meta["limitations"] = [x for x in meta.get("limitations", []) if "legacy V44 balanced reference anchor" not in x]
    meta["limitations"].append("Clean reference lineage only; ETF/REIT/CB cashflow, split and execution contracts still uncertified; no formal training or PPO")
    save("metadata.json", meta)
    final = dict(status="clean_research_reference_built_not_trained", assets=38,
        dataset_sha256=sha(DEST / "dataset.npz"), old_v44_anchor_dependency_removed=True,
        corrected_returns_unchanged=True, nine_grid_unchanged=True,
        first_clean_reference_date=str(aligned.loc[known, "date"].min().date()),
        prefix_checks=len(prefix_checks), prefix_checks_passed=True,
        source_lineage_clean_for_anchor=True, execution_cashflows_certified=False,
        formal_training_allowed=False, ppo_started=False, formal_promotion=False)
    save("status.json", final)
    save("source_hashes.json", {str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)})
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
