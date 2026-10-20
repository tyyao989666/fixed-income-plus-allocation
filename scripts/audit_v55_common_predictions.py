"""Read frozen annual predictions on common masks in raw relative-return units.

Descriptive diagnostics only: never select a candidate or change a gate with OOS data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v55_learning_objective import decode_opportunity

KEY = ["as_of_date", "task", "entity", "module", "horizon_days"]


def attach_truth(frame, arrays, meta, decode):
    frame = frame.copy()
    dates = pd.Index(pd.to_datetime(arrays["dates"], unit="ns"))
    t = dates.get_indexer(pd.to_datetime(frame.as_of_date))
    if (t < 0).any():
        raise ValueError("Forecast date outside frozen input calendar")
    frame["truth"] = np.nan
    frame["observed_label"] = False
    for task, names in [("asset", meta["assets"]), ("sleeve", meta["modules"])]:
        entity_lookup = pd.Index(names)
        for j, h in enumerate([5, 20, 60]):
            take = frame.task.eq(task) & frame.horizon_days.eq(h)
            e = entity_lookup.get_indexer(frame.loc[take, "entity"])
            if (e < 0).any():
                raise ValueError("Unknown entity")
            tt = t[take]
            good = arrays[task + "_target_mask"][tt, j, e]
            raw_good = arrays[task + "_raw_relative_target_mask"][tt, j, e]
            frame.loc[take, "observed_label"] = good & raw_good
            frame.loc[take, "truth"] = arrays[task + "_raw_relative_target"][tt, j, e]
            if decode:
                scale = arrays[task + "_opportunity_scale"][tt, j, e]
                offset = arrays[task + "_opportunity_offset"][tt, j, e]
                for col in ["prediction", "baseline_prediction", "gated_prediction"]:
                    frame.loc[take, col] = decode_opportunity(
                        frame.loc[take, col].to_numpy(), scale, offset)
    if frame.duplicated(KEY + ["model", "seed"]).any():
        raise ValueError("Duplicate prediction keys")
    for col in ["train_end_date", "gate_as_of_date"]:
        if not (pd.to_datetime(frame[col]) < pd.to_datetime(frame.as_of_date)).all():
            raise ValueError("Forecast timestamp isolation failed: " + col)
    return frame


def rank_ic(group):
    if len(group) < 3 or group.truth.nunique() < 2 or group.prediction.nunique() < 2:
        return np.nan
    return group.truth.rank().corr(group.prediction.rank())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v56", action="store_true", help="Compare V55 and V56, both decoded; no fitting")
    v56 = parser.parse_args().v56
    branch = "total_view_rank_v56" if v56 else "temporal_opportunity_v55"
    out = ROOT / "output/reports" / branch
    meta = json.loads((ROOT / "output/reports/temporal_opportunity_v55/metadata.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "data_processed/temporal_opportunity_v55/dataset.npz") as z:
        a = {k: z[k] for k in z.files if "target" in k or "opportunity" in k or k == "dates"}
    metrics, coverage = [], []
    for year in range(2022, 2027):
        roots = ([ROOT / "output/models/temporal_opportunity_v55" / str(year),
                  ROOT / "output/models/total_view_rank_v56" / str(year)] if v56 else
                 [ROOT / "output/models/unified_v53/full" / str(year),
                  ROOT / "output/models/temporal_opportunity_v55" / str(year)])
        if not all((p / "report.json").exists() and (p / "predictions.parquet").exists() for p in roots):
            continue
        frames = []
        for i, root in enumerate(roots):
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            if len(report.get("completed_seeds", [])) != 3:
                break
            f = pd.read_parquet(root / "predictions.parquet", filters=[("seed", "=", -1)])
            f = attach_truth(f, a, meta, decode=(v56 or i == 1))
            # Both arms are scored only where the NEW target is valid too.
            f = f[f.observed_label & np.isfinite(f.truth) & np.isfinite(f.prediction)]
            frames.append(f)
        if len(frames) != 2:
            continue
        for model in ["F0", "F1", "F2", "F3"]:
            pair = [f[f.model.eq(model)].set_index(KEY) for f in frames]
            common = pair[0].index.intersection(pair[1].index)
            pair = [f.loc[common].reset_index() for f in pair]
            if not np.allclose(pair[0].truth, pair[1].truth, atol=0, rtol=0):
                raise ValueError("Different truths on common keys")
            arms = ["V55_decoded_return", "V56_decoded_return"] if v56 else ["V53_raw_return", "V55_decoded_return"]
            for arm, f in zip(arms, pair):
                for (task, module, horizon), g in f.groupby(["task", "module", "horizon_days"]):
                    daily_ic = pd.Series([rank_ic(day) for _, day in g.groupby("as_of_date")])
                    dispersions = g.groupby("as_of_date").prediction.std(ddof=0)
                    metrics.append(dict(arm=arm, year=year, model=model, task=task,
                        module=module, horizon_days=int(horizon), cells=len(g),
                        observed_dates=g.as_of_date.nunique(), rank_ic_dates=int(daily_ic.notna().sum()),
                        mean_daily_cross_section_rank_ic=daily_ic.mean(),
                        raw_return_mae=float(np.abs(g.prediction-g.truth).mean()),
                        mean_cross_section_prediction_std=dispersions.mean(),
                        gated_equals_baseline=bool(np.allclose(g.gated_prediction, g.baseline_prediction)),
                        active_date_fraction=float(g.active_component.mean())))
                coverage.append(dict(arm=arm, year=year, model=model, common_cells=len(f),
                                     timestamp_checks_passed=True))
    pd.DataFrame(metrics).to_csv(out / "common_raw_return_diagnostics.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "common_prediction_coverage.csv", index=False)
    result = dict(completed_common_years=sorted({r["year"] for r in coverage}), branch=branch,
                  rows=len(metrics), units="raw_unannualized_relative_return",
                  model_or_parameter_selection_performed=False,
                  caveat="Cross-sectional rank IC within modules is not market timing or portfolio Sharpe; sleeve singleton IC undefined.",
                  formal_portfolio_promotion=False)
    (out / "common_prediction_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
