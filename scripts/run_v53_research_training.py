"""Hash-bound mask-safe forecast research; never authorizes portfolio deployment."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation, source_hash
from src.v53_unified_data import validate_relative_dataset
from scripts.run_v52_execution_preflight import save, sha
from scripts.run_v45_research_pipeline import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Real 2026 fold, seed 17, unchanged 1000 epoch ceilings; never full promotion")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset_path = ROOT / "data_processed/unified_v53/dataset.npz"
    report = ROOT / "output/reports/unified_v53"
    metadata_path = report / "metadata.json"
    state = json.loads((report / "status.json").read_text(encoding="utf-8"))
    if state["dataset_sha256"] != sha(dataset_path) or state["metadata_sha256"] != sha(metadata_path):
        raise ValueError("Dataset/metadata changed since registration")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for entry in metadata["source_manifest"].values():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Source changed: " + entry["path"])
    with np.load(dataset_path, allow_pickle=False) as f:
        arrays = {k:f[k].copy() for k in f.files}
    validate_relative_dataset(arrays)
    cfg = engine.configuration(ROOT / "config/v45_research.yaml")
    cfg["seed_workers"] = 1  # Mask-safe network must survive the actual worker path.
    cfg["v53_implementation"] = dict(mask_safe_hash=source_hash(), dataset_sha256=sha(dataset_path),
        scope="research_forecast_only_no_portfolio_cashflow_certification", formal_promotion=False,
        data_builder_sha256=sha(ROOT / "scripts/build_v53_unified_dataset.py"))
    if args.pilot:
        cfg["years"], cfg["seeds"] = [2026], [17]
    run_dir = report / ("pilot_forecast" if args.pilot else "forecast")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = report / "active_forecast_training.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if process_alive(lock["pid"]):
            raise RuntimeError("A V53 forecast fit is already active; inspect progress, do not launch a duplicate")
    started = time.monotonic()
    progress = dict(status="preflight", pilot=args.pilot, training_started=False,
        full_three_seed_retraining=False, portfolio_training_started=False, formal_promotion=False,
        dataset_sha256=sha(dataset_path), mask_safe_source_hash=source_hash(),
        max_epochs=cfg["max_epochs"], pretrain_max_epochs=cfg["pretrain_max_epochs"])
    save(run_dir / "training_status.json", progress)
    preflight = engine.forecast_preflight(arrays, cfg)
    engine.write_preflight(run_dir, preflight)
    # Refuse to fit a fold whose V-selection is structurally undefined; changing
    # gates/horizon weights to conceal insufficient labels is not permitted.
    if not all(f["structural_selection_available"] for f in preflight["folds"]):
        progress.update(status="blocked_structural_validation", elapsed_seconds=round(time.monotonic()-started,1))
        save(run_dir / "training_status.json", progress)
        print(json.dumps(progress, indent=2))
        return 2
    if not args.train:
        progress.update(status="preflight_passed", elapsed_seconds=round(time.monotonic()-started,1))
        save(run_dir / "training_status.json", progress)
        print(json.dumps(progress, indent=2))
        return 0
    progress.update(status="training", training_started=True)
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid(), pilot=args.pilot, dataset_sha256=sha(dataset_path)), stream)
    except FileExistsError:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if process_alive(lock["pid"]):
            raise RuntimeError("Concurrent V53 training refused")
        lock_path.unlink()
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid(), pilot=args.pilot, dataset_sha256=sha(dataset_path)), stream)
    save(run_dir / "training_status.json", progress)
    try:
        with training_implementation():
            result = engine.run_training(arrays, metadata, cfg,
                ROOT / "output/models/unified_v53" / ("pilot" if args.pilot else "full"),
                ROOT / "data_processed/unified_v53" / ("pilot_predictions.parquet" if args.pilot else "predictions.parquet"),
                run_dir, resume=args.resume, debug=args.pilot)
        progress.update(status="completed_research_forecast", full_three_seed_retraining=not args.pilot,
            elapsed_seconds=round(time.monotonic()-started,1),
            forecast_report=str((run_dir / "v45_forecast_report.json").relative_to(ROOT)))
        save(run_dir / "training_status.json", progress)
        latest = json.loads((report / "status.json").read_text(encoding="utf-8"))
        latest.update(research_pilot_trained=bool(args.pilot), full_forecast_retrained=not args.pilot,
            models_retrained=not args.pilot, latest_forecast_status=str((run_dir / "training_status.json").relative_to(ROOT)))
        save(report / "status.json", latest)
        print(json.dumps(progress, ensure_ascii=False, indent=2))
    except BaseException as exc:
        progress.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=round(time.monotonic()-started,1))
        save(run_dir / "training_status.json", progress)
        raise
    finally:
        if lock_path.exists():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if lock["pid"] == os.getpid():
                lock_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
