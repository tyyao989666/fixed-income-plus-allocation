from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_command(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def read_csv_if_exists(path: str) -> list[dict[str, object]]:
    full_path = ROOT / path
    if not full_path.exists():
        return []
    return pd.read_csv(full_path).to_dict(orient="records")


def write_manifest(skip_baseline: bool, skip_itransformer: bool) -> None:
    manifest = {
        "pipeline": "v2_itransformer_signal_pipeline",
        "scope": "signal_only_not_allocator",
        "deep_aries_reference": {
            "used": [
                "inverted time-series tokenization",
                "Transformer forecasting signal",
                "future PPO as constrained tilt/rebalance controller",
            ],
            "not_used": [
                "direct stock top-k allocation",
                "unconstrained PPO portfolio weights",
                "daily 1/5/20-day horizon environment",
            ],
        },
        "outputs": {
            "dataset": "data_processed/itransformer_dataset_monthly.parquet",
            "metadata": "output/reports/itransformer_dataset_metadata.json",
            "baseline_predictions": "data_processed/baseline_predictions_monthly.parquet",
            "baseline_report": "output/reports/baseline_prediction_report.csv",
            "itransformer_model": "data_processed/itransformer_signal_model.pt",
            "itransformer_predictions": "data_processed/itransformer_predictions_monthly.parquet",
            "itransformer_report": "output/reports/itransformer_validation_report.csv",
            "training_log": "output/reports/itransformer_training_log.csv",
            "visual_report": "output/reports/itransformer_signal_report.html",
            "multiseed_report": "output/reports/itransformer_multiseed_report.csv",
            "multiseed_summary": "output/reports/itransformer_multiseed_summary.csv",
        },
        "baseline_report": [] if skip_baseline else read_csv_if_exists("output/reports/baseline_prediction_report.csv"),
        "itransformer_report": [] if skip_itransformer else read_csv_if_exists("output/reports/itransformer_validation_report.csv"),
    }
    path = ROOT / "output" / "reports" / "v2_signal_pipeline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v2 iTransformer signal-only pipeline.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip rolling-mean baseline predictions.")
    parser.add_argument("--skip-itransformer", action="store_true", help="Skip iTransformer training and prediction.")
    parser.add_argument("--with-multiseed", action="store_true", help="Run configured multi-seed iTransformer stability checks.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip iTransformer report figures and HTML.")
    args = parser.parse_args()

    run_command("build iTransformer monthly dataset", [PYTHON, "scripts/07_build_itransformer_dataset.py"])
    if not args.skip_baseline:
        run_command("run rolling baseline", [PYTHON, "scripts/08_train_baseline_models.py"])
    if not args.skip_itransformer:
        run_command("train iTransformer signal model", [PYTHON, "scripts/09_train_itransformer.py"])
    if args.with_multiseed:
        run_command("run iTransformer multi-seed checks", [PYTHON, "scripts/54_train_itransformer_multiseed.py"])
    if not args.skip_visuals:
        run_command("generate iTransformer visual report", [PYTHON, "scripts/55_visualize_itransformer_results.py"])

    write_manifest(args.skip_baseline, args.skip_itransformer)
    print("\n=== done ===", flush=True)
    print("Manifest: output/reports/v2_signal_pipeline_manifest.json", flush=True)


if __name__ == "__main__":
    main()
