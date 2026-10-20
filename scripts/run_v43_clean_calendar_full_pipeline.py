"""Run the reproducible 38-asset, clean-calendar research release.

The runner deliberately does not refresh external vendors.  Vendor refreshes can
be partial or delayed; mixing them into a historical walk-forward run would make
the result irreproducible.  The formal release is therefore locked at the
configured final date and every downstream stage is resumable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config/hierarchical_asset_itransformer_v43_clean_calendar.yaml"
MANIFEST = ROOT / "output/reports/v43_clean_calendar_full_pipeline_manifest.json"


def run_stage(name: str, args: list[str], manifest: list[dict[str, object]]) -> None:
    started = datetime.now().isoformat(timespec="seconds")
    print(f"\n=== {name} ===", flush=True)
    completed = subprocess.run([sys.executable, *args], cwd=ROOT, check=False)
    manifest.append(
        {
            "stage": name,
            "command": [sys.executable, *args],
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "returncode": completed.returncode,
        }
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if completed.returncode:
        raise SystemExit(f"Stage failed: {name} (exit {completed.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-stage",
        choices=["data", "dataset", "baseline", "train", "probe", "portfolio", "plots"],
        default="data",
        help="Resume from this stage. Earlier completed stages are left untouched.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--through-stage",
        choices=["data", "dataset", "baseline", "train", "probe", "portfolio", "plots"],
        default="plots",
        help="Stop after this stage; useful for validating or preparing a long resumable run.",
    )
    args = parser.parse_args()

    stages = [
        ("data", "build_v43_basic_data", ["scripts/47_build_v43_basic_data.py"]),
        ("data", "audit_v43_no_annual_data", ["scripts/48_audit_v43_no_annual_data.py"]),
        ("data", "generate_market_state_cards", ["scripts/10_generate_market_state_cards.py"]),
        ("data", "build_mechanical_ep_views", ["scripts/12_build_ep_views.py", "--source", "mechanical"]),
        ("dataset", "build_38_asset_clean_calendar_dataset", ["scripts/134_build_hierarchical_asset_itransformer_v7.py", "--config", CONFIG]),
        ("baseline", "run_A0_A1_control_variable_baselines", ["scripts/171_run_v43_control_variable_baselines.py"]),
        ("train", "train_itransformer_walk_forward", ["scripts/135_train_hierarchical_asset_itransformer_v7.py", "--config", CONFIG, "--resume"]),
        ("probe", "train_causal_relative_view_probe", ["scripts/163_train_causal_linear_probe_itransformer.py", "--config", CONFIG, "--resume"]),
        ("portfolio", "run_ep_skfolio_teacher_ppo", ["scripts/136_run_hierarchical_asset_itransformer_ppo_v7.py", "--config", CONFIG, "--profiles", "low_vol,balanced,high_vol"]),
        ("plots", "plot_training", ["scripts/138_plot_hierarchical_asset_itransformer_v7_training.py", "--config", CONFIG]),
    ]
    allowed = ["data", "dataset", "baseline", "train", "probe", "portfolio", "plots"]
    start_index = allowed.index(args.from_stage)
    end_index = allowed.index(args.through_stage)
    if end_index < start_index:
        raise SystemExit("--through-stage must not be earlier than --from-stage")
    manifest: list[dict[str, object]] = []
    for category, name, command in stages:
        category_index = allowed.index(category)
        if (
            category_index < start_index
            or category_index > end_index
            or (args.skip_plots and category == "plots")
        ):
            continue
        run_stage(name, command, manifest)

    print(f"\n=== done ===\nManifest: {MANIFEST.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
