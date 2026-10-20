"""One-command source -> dataset -> checks -> forecast research, no silent promotion."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true", help="Actual research forecast training after data checks")
    p.add_argument("--pilot", action="store_true", help="2026/seed17 only, not full three-seed research")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(ROOT))
    from scripts.run_v45_research_pipeline import process_alive
    lock = ROOT / "output/reports/unified_v53/active_forecast_training.json"
    if lock.exists() and process_alive(json.loads(lock.read_text(encoding="utf-8"))["pid"]):
        raise SystemExit("Forecast fit is active; refusing to rebuild or replace its input dataset")
    launch = ROOT / "output/reports/unified_v53/full_training_launch.json"
    if launch.exists() and process_alive(json.loads(launch.read_text(encoding="utf-8"))["pid"]):
        raise SystemExit("Full forecast process is active, including preflight; refusing input rebuild")
    stages = [["scripts/acquire_v53_unified_sources.py"], ["scripts/test_v53_unified_data.py"],
              ["scripts/build_v53_unified_dataset.py"], ["scripts/audit_v53_unified_dataset.py"]]
    training = ["scripts/run_v53_research_training.py"]
    if args.train:
        training.append("--train")
    if args.pilot:
        training.append("--pilot")
    if args.resume:
        training.append("--resume")
    stages.append(training)
    for command in stages:
        print("V53 stage: " + " ".join(command), flush=True)
        subprocess.run([sys.executable, "-u", "-X", "utf8", *command], cwd=ROOT, check=True)
    print("Research stages finished. This runner does NOT start uncertified PPO/portfolio execution.", flush=True)


if __name__ == "__main__":
    main()
