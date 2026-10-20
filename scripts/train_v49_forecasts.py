"""Explicit opt-in mask-safe research training, fail closed on unresolved data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src import v45_forecast as engine
from src.v49_masked_forecast import validate_dataset, training_implementation, source_hash


def require_ready(readiness):
    if readiness.get("training_ready") is not True or readiness.get("blockers"):
        raise ValueError("Data reconstruction is not approved: "+"; ".join(readiness.get("blockers", ["training_ready is not true"])))
    if readiness.get("cutoff") != "2026-10-14":
        raise ValueError("Unexpected requested data cutoff")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT/"data_processed/v49_dataset.npz")
    parser.add_argument("--metadata", type=Path, default=ROOT/"output/reports/v49_metadata.json")
    parser.add_argument("--readiness", type=Path, default=ROOT/"output/reports/v49_project_review/data_readiness.json")
    parser.add_argument("--train", action="store_true", help="Train only after data readiness approval; default only checks")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    try:
        require_ready(readiness)
    except ValueError as exc:
        engine.write_json(ROOT/"output/reports/v49_project_review/training_preflight.json",
                          dict(allowed=False, reason=str(exc), new_models_trained=False, formal_promotion=False))
        print(str(exc), flush=True)
        return 2
    with np.load(args.dataset, allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    validate_dataset(arrays)
    if (len(metadata["assets"]) != 38 or len(set(metadata["assets"])) != 38
            or len(arrays["module_index"]) != 38 or metadata.get("end") != "2026-10-14"
            or str(engine.pd.Timestamp(arrays["dates"][-1]).date()) != "2026-10-14"):
        raise ValueError("Expected 38-context-asset dataset through requested cutoff")
    if metadata.get("historical_CB_reconstruction_verified") is not True:
        raise ValueError("Refusing a splice of repaired CB buckets onto unverified old history")
    if readiness.get("dataset_sha256") != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
        raise ValueError("Dataset hash is not bound to the approval")
    if readiness.get("metadata_sha256") != hashlib.sha256(args.metadata.read_bytes()).hexdigest():
        raise ValueError("Metadata hash is not bound to the approval")
    cfg = copy.deepcopy(engine.configuration(ROOT/"config/v45_research.yaml"))
    cfg["seed_workers"] = 1
    cfg["v49_implementation"] = dict(mask_semantics="masked_assets_excluded_from_cross_attention_and_sleeve_pool",
                                     source_hash=source_hash(), new_weights_required=True)
    engine.write_json(ROOT/"output/reports/v49_project_review/training_preflight.json",
                      dict(allowed=True, train_requested=args.train, formal_promotion=False,
                           protocol="same annual F0-F3 T/V/G research; NOT a newly designed October issuance model"))
    if args.train:
        with training_implementation():
            engine.run_training(arrays, metadata, cfg, ROOT/"output/models/v49_forecast",
                ROOT/"data_processed/v49_predictions.parquet", ROOT/"output/reports/v49_forecast",
                resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
