"""Stream-check actual completed score/return predictions without fitting anything."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_v52_execution_preflight import sha
from src.v55_learning_objective import decode_opportunity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["temporal_opportunity_v55", "total_view_rank_v56"],
                        default="temporal_opportunity_v55")
    branch = parser.parse_args().branch
    reports = ROOT / "output/reports" / branch
    data = ROOT / "data_processed" / branch
    parent = ROOT / "output/reports/temporal_opportunity_v55"
    input_data = ROOT / "data_processed/temporal_opportunity_v55"
    status = json.loads((reports / "training_status.json").read_text())
    assert status["status"] == "completed_objective_research"
    meta = json.loads((parent / "metadata.json").read_text())
    assert sha(input_data / "dataset.npz") == status["dataset_sha256"]
    if branch == "total_view_rank_v56":
        implementation = status["experiment"]
        for path, key in [("src/v56_total_view_ranking.py", "ranking_sha256"),
                          ("scripts/run_v56_total_view_ranking.py", "runner_sha256"),
                          ("docs/v56_total_view_ranking_preregistration.md", "preregistration_sha256")]:
            assert sha(ROOT / path) == implementation[key]
    for source in meta["source_manifest"].values():
        assert sha(ROOT / source["path"]) == source["sha256"]
    with np.load(input_data / "dataset.npz") as z:
        a = {k: z[k] for k in z.files if "opportunity" in k or "target_mask" in k or k == "dates"}
    dates = pd.Index(pd.to_datetime(a["dates"], unit="ns"))
    score = pq.ParquetFile(data / "score_predictions.parquet")
    decoded = pq.ParquetFile(data / "decoded_relative_return_views.parquet")
    assert score.metadata.num_rows == decoded.metadata.num_rows
    count, maximum, nonfinite, active = 0, 0., 0, 0
    seed_rows, prediction_checks = [], []
    for sb, db in zip(score.iter_batches(batch_size=50000), decoded.iter_batches(batch_size=50000)):
        s, d = sb.to_pandas(), db.to_pandas()
        for key in ["as_of_date", "task", "entity", "horizon_days", "model", "seed", "model_signature"]:
            assert s[key].equals(d[key]), key
        assert d.prediction_units.eq("raw_unannualized_relative_return").all()
        t = dates.get_indexer(pd.to_datetime(s.as_of_date))
        assert (t >= 0).all()
        for task, names in [("asset", meta["assets"]), ("sleeve", meta["modules"])]:
            for j, h in enumerate([5, 20, 60]):
                take = s.task.eq(task) & s.horizon_days.eq(h)
                e = pd.Index(names).get_indexer(s.loc[take, "entity"])
                assert (e >= 0).all()
                tt = t[take]
                scale = a[task+"_opportunity_scale"][tt, j, e]
                offset = a[task+"_opportunity_offset"][tt, j, e]
                for col in ["prediction", "baseline_prediction", "gated_prediction", "raw_increment", "active_increment"]:
                    value = s.loc[take, col].to_numpy()
                    expected = (decode_opportunity(value, scale, offset) if col.endswith("prediction")
                                else np.where(scale > 0, value*scale, np.nan))
                    actual = d.loc[take, col].to_numpy()
                    assert np.allclose(expected, actual, rtol=1e-6, atol=1e-8, equal_nan=True), col
                    finite = np.isfinite(expected) & np.isfinite(actual)
                    if finite.any():
                        maximum = max(maximum, float(np.abs(expected[finite]-actual[finite]).max()))
        nonfinite += int((~np.isfinite(d.prediction)).sum())
        active += int(d.active_component.sum())
        for key in ["train_end_date", "gate_as_of_date"]:
            assert (pd.to_datetime(d[key]) < pd.to_datetime(d.as_of_date)).all()
        count += len(d)
    assert count == score.metadata.num_rows
    for root in sorted((ROOT / "output/models" / branch).glob("20*")):
        path = root / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        assert report["completed_seeds"] == [17, 29, 47]
        f = pd.read_parquet(root / "predictions.parquet")
        for model in ["F1", "F2", "F3"]:
            keys = ["as_of_date", "task", "entity", "module", "horizon_days"]
            wide = f[f.model.eq(model) & f.seed.ne(-1)].pivot(index=keys, columns="seed", values="prediction")
            assert list(wide.columns) == [17, 29, 47]
            full = wide.dropna()
            assert len(full) > 0
            seed_rows.append(dict(year=int(root.name), model=model, complete_seed_cells=len(full),
                                  mean_seed_std_score_units=float(full.std(axis=1, ddof=0).mean()),
                                  corr_17_29=float(full[17].corr(full[29])),
                                  corr_17_47=float(full[17].corr(full[47])),
                                  corr_29_47=float(full[29].corr(full[47]))))
        prediction_checks.append(dict(year=int(root.name), gates=len(report["gates"]),
                                      active_gates=sum(bool(g["active"]) for g in report["gates"])))
    for task in ["asset", "sleeve"]:
        for j, h in enumerate([5, 20, 60]):
            assert not a[task+"_target_mask"][-h:, j].any()
            assert not a[task+"_raw_relative_target_mask"][-h:, j].any()
    pd.DataFrame(seed_rows).to_csv(reports / "final_seed_dispersion.csv", index=False)
    result = dict(passed=True, branch=branch, rows=count, decoded_max_absolute_error=maximum,
                  missing_forecasts=nonfinite, active_prediction_rows=active,
                  annual_gates=prediction_checks, source_hashes_passed=True,
                  immature_label_tails_masked=True, timestamp_isolation_passed=True,
                  seed_correlations_are_descriptive_not_gate_overrides=True,
                  formal_promotion=False, ppo_started=False)
    (reports / "final_decode_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
