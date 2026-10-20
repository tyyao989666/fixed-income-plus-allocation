"""Train frozen-data v45 F0/F1/F2/F3; no portfolio or Agentic side effects."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v45_forecast import (configuration, run_training, shared_split,
    synthetic_arrays, throughput_probe, write_json, forecast_preflight,
    write_preflight, require_selectable_preflight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v45_research.yaml")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data_processed/v45_dataset.npz")
    parser.add_argument("--metadata", type=Path, default=ROOT / "output/reports/v45_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/models/v45_forecast")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data_processed/v45_predictions.parquet")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "output/reports")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, choices=[1, 2, 4], default=None)
    parser.add_argument("--seed-workers", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--epochs", type=int, help="Explicit DEBUG cap for BOTH training stages; never formal")
    parser.add_argument("--years", help="Explicit DEBUG subset, comma-separated; never formal")
    parser.add_argument("--probe", "--benchmark-only", action="store_true", help="Throughput benchmark only; no full fits")
    parser.add_argument("--synthetic-smoke", "--smoke", action="store_true", help="Tiny synthetic fits in a temporary directory only")
    parser.add_argument("--dump-config", action="store_true", help="Print resolved configuration and exit")
    parser.add_argument("--preflight-only", action="store_true", help="Print actual F0/OOF T/V coverage; no neural training or G evaluation")
    args = parser.parse_args()
    cfg = configuration(args.config)
    if args.threads:
        cfg["threads"] = args.threads
    if args.seed_workers:
        cfg["seed_workers"] = args.seed_workers
    debug = args.epochs is not None or args.years is not None or args.synthetic_smoke
    if args.epochs is not None:
        if not 1 <= args.epochs <= 1000:
            parser.error("--epochs must be between 1 and 1000")
        # Do not silently lower minimum epochs; an explicit debug budget may end
        # before them, is clearly reported and cannot release a component.
        cfg["max_epochs"] = cfg["pretrain_max_epochs"] = args.epochs
    if args.years:
        cfg["years"] = sorted(set(int(x) for x in args.years.split(",")))
    if args.dump_config:
        print(json.dumps(dict(config=cfg, debug=debug), indent=2))
        return
    if args.synthetic_smoke:
        from src.v45_forecast import train_network, ForecastNetwork, quarterly_oof, raw_features, GroupRidge, targets_at, residual_scaling, infer
        a = synthetic_arrays(rows=1100)
        cfg = copy.deepcopy(cfg)
        cfg["max_epochs"] = cfg["pretrain_max_epochs"] = args.epochs or 2
        cfg["pretrain_batch"] = 8
        cfg["supervised_batch"] = 8
        cfg["threads"] = args.threads or 2
        cfg["seed_workers"] = 1
        torch.set_num_threads(cfg["threads"])
        split = shared_split(a, 2021, cfg)
        # Small subsets are explicit synthetic-only; the production splitter and
        # 100/50 minimum-epoch settings are unchanged.
        split["train"] = split["train"][-24:]
        split["validation"] = split["validation"][:40]
        device = torch.device(args.device or "cpu")
        with tempfile.TemporaryDirectory(prefix="v45_synthetic_") as tmp:
            network, summary = train_network(ForecastNetwork(a, cfg), a, split, cfg,
                device, Path(tmp) / "pretrain.pt", "synthetic_only", "pretraining", 17)
            y, mask = targets_at(a, split["train"])
            # OOF is generated on the full honest T before taking the small
            # optimizer fixture, never using an in-sample baseline residual.
            full_train = shared_split(a, 2021, cfg)["train"]
            oof, audit = quarterly_oof(a, full_train, cfg)
            positions = np.searchsorted(full_train, split["train"])
            oof = {t: values[positions] for t, values in oof.items()}
            baseline = GroupRidge(a["module_index"]).fit(raw_features(a, full_train), *targets_at(a, full_train))
            vb = baseline.predict(raw_features(a, split["validation"]))
            residual, good, center, scale = residual_scaling(y, oof, mask, a["module_index"])
            representations = infer(network, a, split["train"], cfg, device, True)
            probe = GroupRidge(a["module_index"]).fit(representations, residual, good)
            assert probe.models, "No OOF residual fixture available"
            stages = [summary]
            for candidate in ("F2", "F3"):
                model, result = train_network(copy.deepcopy(network), a, split, cfg, device,
                    Path(tmp) / f"{candidate}.pt", "synthetic_only", candidate, 17,
                    residual, good, center, scale, vb)
                assert np.isfinite(infer(model, a, split["validation"][:4], cfg, device)["asset"]).all()
                stages.append(result)
            result = dict(synthetic_only=True, stages=stages, quarterly_oof_blocks=len(audit),
                throughput=throughput_probe(a, cfg, device))
            print(json.dumps(result, indent=2))
        return
    with np.load(args.dataset, allow_pickle=False) as dataset:
        a = {key: dataset[key] for key in dataset.files}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if args.preflight_only:
        result = forecast_preflight(a, cfg)
        write_preflight(args.reports_dir, result)
        print(json.dumps({"folds": [{k: v for k, v in fold.items() if k != "components"}
            for fold in result["folds"]], "gate_performance_opened": False}, indent=2))
        require_selectable_preflight(result)
        return
    if args.probe:
        probe = throughput_probe(a, cfg, args.device)
        estimates = []
        for year in cfg["years"]:
            split = shared_split(a, year, cfg)
            train_count = len(split["train"])
            batches = int(np.ceil(train_count/cfg["pretrain_batch"]))
            epoch_seconds = batches*probe["seconds_per_pretrain_batch"]
            estimates.append(dict(year=year, train_dates=train_count,
                pretrain_epoch_seconds=epoch_seconds,
                three_seed_pretrain_minimum_hours=3*cfg["pretrain_minimum_epochs"]*epoch_seconds/3600,
                three_seed_pretrain_maximum_hours=3*cfg["pretrain_max_epochs"]*epoch_seconds/3600,
                excludes=probe["excludes"]))
        report = dict(probe=probe, estimates=estimates, config=cfg)
        write_json(args.reports_dir / "v45_forecast_throughput.json", report)
        print(json.dumps(report, indent=2))
        return
    report = run_training(a, metadata, cfg, args.output_dir, args.predictions,
        args.reports_dir, args.resume, debug, args.device)
    print(json.dumps({key: value for key, value in report.items() if key not in ("folds", "zero_increment_trace")}, indent=2))


if __name__ == "__main__":
    main()
