"""Separate missing inverse parameters from missing scores; audit neural increments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["temporal_opportunity_v55", "total_view_rank_v56"],
                        default="temporal_opportunity_v55")
    branch = parser.parse_args().branch
    report = ROOT / "output/reports" / branch
    meta = json.loads((ROOT / "output/reports/temporal_opportunity_v55/metadata.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "data_processed/temporal_opportunity_v55/dataset.npz") as z:
        dates = pd.Index(pd.to_datetime(z["dates"], unit="ns"))
        scales = {t: z[t+"_opportunity_scale"] for t in ["asset", "sleeve"]}
    score = pd.read_parquet(ROOT / "data_processed" / branch / "score_predictions.parquet",
                            filters=[("seed", "=", -1), ("model", "=", "F0")])
    score["missing_score"] = ~np.isfinite(score.prediction)
    score["missing_inverse_parameters"] = False
    indices = dates.get_indexer(pd.to_datetime(score.as_of_date))
    assert (indices >= 0).all()
    for task, names in [("asset", meta["assets"]), ("sleeve", meta["modules"])]:
        for j, h in enumerate([5, 20, 60]):
            take = score.task.eq(task) & score.horizon_days.eq(h)
            e = pd.Index(names).get_indexer(score.loc[take, "entity"])
            assert (e >= 0).all()
            risk = scales[task][indices[take], j, e]
            score.loc[take, "missing_inverse_parameters"] = ~np.isfinite(risk) | (risk <= 0)
    missing = score[score.missing_score | score.missing_inverse_parameters].copy()
    missing["reason"] = np.select(
        [missing.missing_score & missing.missing_inverse_parameters, missing.missing_score],
        ["score_and_inverse_parameters_unavailable", "score_unavailable_inverse_parameters_available"],
        default="inverse_parameters_unavailable_score_available")
    coverage = missing.groupby(["task", "entity", "horizon_days", "reason"]).agg(
        cells=("as_of_date", "size"), first_date=("as_of_date", "min"),
        last_date=("as_of_date", "max")).reset_index()
    coverage.to_csv(report / "missing_prediction_causes.csv", index=False)

    consistency = []
    for path in sorted((ROOT / "output/models" / branch).glob("20*/predictions.parquet")):
        f = pd.read_parquet(path, filters=[("seed", "!=", -1)])
        for (model, task, module, h), g in f.groupby(["model", "task", "module", "horizon_days"]):
            wide = g.pivot(index=["as_of_date", "entity"], columns="seed", values="raw_increment").dropna()
            assert list(wide.columns) == [17, 29, 47]
            correlations = []
            for i, j in [(17,29), (17,47), (29,47)]:
                correlations.append(float(wide[i].corr(wide[j]))
                                    if len(wide) > 2 and wide[i].std() > 0 and wide[j].std() > 0 else np.nan)
            signs = np.sign(wide.to_numpy())
            consistency.append(dict(year=int(path.parent.name), model=model, task=task,
                module=module, horizon_days=int(h), cells=len(wide),
                mean_seed_std_increment_units=float(wide.std(axis=1, ddof=0).mean()),
                mean_abs_increment=float(wide.abs().to_numpy().mean()),
                all_seed_sign_agreement=float((signs == signs[:, :1]).all(axis=1).mean()),
                corr_17_29=correlations[0], corr_17_47=correlations[1], corr_29_47=correlations[2]))
    pd.DataFrame(consistency).to_csv(report / "neural_increment_seed_consistency.csv", index=False)
    summary = dict(unique_F0_ensemble_missing_cells=len(missing),
        causes=missing.reason.value_counts().to_dict(),
        increment_consistency_rows=len(consistency),
        units="V55 dimensionless opportunity increments; decoded returns not used here",
        caveat="Descriptive pooled-date/entity correlations, not independent observations or gate overrides",
        no_imputation=True, formal_promotion=False)
    (report / "missing_and_increment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
