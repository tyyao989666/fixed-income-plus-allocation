"""One preregistered GPU experiment: align F3 total-view ranking and selection."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation, source_hash
from src.v56_total_view_ranking import total_view_ranking_implementation
from scripts.run_v55_gpu_objective import honest_score_report_units, decode_predictions
from scripts.run_v52_execution_preflight import save, sha
from scripts.run_v45_research_pipeline import process_alive


def require_completed_controls():
    for name, root in [("CPU", ROOT / "output/models/unified_v53/full"),
                       ("V55", ROOT / "output/models/temporal_opportunity_v55")]:
        for year in range(2022, 2027):
            path = root / str(year) / "report.json"
            if not path.exists() or json.loads(path.read_text())["completed_seeds"] != [17, 29, 47]:
                raise RuntimeError(name + " control not yet complete")
    for path in [ROOT / "output/reports/unified_v53/full_training_launch.json",
                 ROOT / "output/reports/temporal_opportunity_v55/GPU_training_launch.json"]:
        if process_alive(json.loads(path.read_text(encoding="utf-8-sig"))["pid"]):
            raise RuntimeError("Prior control process still active; wait without duplicating work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V56 requires the isolated CUDA interpreter")
    report = ROOT / "output/reports/total_view_rank_v56"
    output = ROOT / "data_processed/total_view_rank_v56"
    parent = ROOT / "output/reports/temporal_opportunity_v55"
    registration = json.loads((parent / "status.json").read_text())
    audit = json.loads((parent / "final_decode_audit.json").read_text())
    if not audit["passed"]:
        raise RuntimeError("V55 final decode audit must pass first")
    path = ROOT / "data_processed/temporal_opportunity_v55/dataset.npz"
    assert sha(path) == registration["dataset_sha256"]
    assert sha(parent / "metadata.json") == registration["metadata_sha256"]
    meta = json.loads((parent / "metadata.json").read_text())
    for source in meta["source_manifest"].values():
        assert sha(ROOT / source["path"]) == source["sha256"]
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    cfg = engine.configuration(ROOT / "config/v45_research.yaml")
    cfg["seed_workers"] = 1
    cfg["v56_implementation"] = dict(
        dataset_sha256=registration["dataset_sha256"], mask_safe_hash=source_hash(),
        ranking_sha256=sha(ROOT / "src/v56_total_view_ranking.py"),
        runner_sha256=sha(Path(__file__)),
        preregistration_sha256=sha(ROOT / "docs/v56_total_view_ranking_preregistration.md"),
        device="cuda", only_learning_change="F3 rank full OOF total view instead of residual",
        budget="one experiment; fixed original seeds/years/gates; no test-tuned weights")
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    preflight = engine.forecast_preflight(a, cfg)
    engine.write_preflight(report, preflight)
    if any(not fold["structural_selection_available"] for fold in preflight["folds"]):
        raise RuntimeError("Original structural selection requirements unmet")
    if not args.train:
        print("V56 preflight passed; training not started", flush=True)
        return 0
    require_completed_controls()
    lock = report / "active_training.json"
    if lock.exists() and process_alive(json.loads(lock.read_text())["pid"]):
        raise RuntimeError("V56 already running")
    state = dict(status="training", pid=os.getpid(), training_started=True,
                 cuda_device=torch.cuda.get_device_name(0), formal_promotion=False,
                 ppo_started=False, dataset_sha256=registration["dataset_sha256"],
                 experiment=cfg["v56_implementation"])
    save(lock, dict(pid=os.getpid()))
    save(report / "training_status.json", state)
    try:
        with training_implementation(), total_view_ranking_implementation(engine), honest_score_report_units():
            engine.run_training(a, meta, cfg, ROOT / "output/models/total_view_rank_v56",
                                output / "score_predictions.parquet", report,
                                resume=args.resume, debug=False, device="cuda")
        decode_predictions(a, meta, output / "score_predictions.parquet",
                           output / "decoded_relative_return_views.parquet")
        state.update(status="completed_objective_research", decoded_views_available=True)
        save(report / "training_status.json", state)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(report / "training_status.json", state)
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid():
            lock.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
