"""One-entry review: optional acquisition, tests, data build, real model audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v45_forecast import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing supplementary source responses; use cache otherwise")
    args = parser.parse_args()
    folder = ROOT/"output/reports/v49_project_review"
    folder.mkdir(parents=True, exist_ok=True)
    progress = folder/"review_pipeline.json"
    steps = (["supplement_v49_source_data.py"] if args.fetch else [])+[
        "test_stage_20261014_market_data.py", "test_v49_point_in_time.py",
        "test_v49_masked_forecast.py", "build_v49_verified_inputs.py", "audit_v49_forecast_integrity.py"]
    completed = []
    for step in steps:
        write_json(progress, dict(status="running", pid=os.getpid(), current=step, completed=completed,
                                  formal_promotion=False))
        print("[review] "+step, flush=True)
        with (folder/(step.removesuffix(".py")+".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-X", "utf8", "-u", str(ROOT/"scripts"/step)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(progress, dict(status="failed", failed_stage=step, returncode=result.returncode,
                                      completed=completed, formal_promotion=False))
            raise RuntimeError(f"Review failed: {step}; inspect its dedicated log")
        completed.append(step)
    readiness = json.loads((folder/"data_readiness.json").read_text(encoding="utf-8"))
    integrity = json.loads((folder/"forecast_integrity_audit.json").read_text(encoding="utf-8"))
    write_json(progress, dict(status="review_completed_data_reconstruction_blocked" if not readiness["training_ready"] else "review_completed",
        completed=completed, engineering_checks_passed=integrity["engineering_checks_passed"],
        training_ready=readiness["training_ready"], blockers=readiness["blockers"],
        formal_promotion=False, old_results_overwritten=False))
    print("Review complete. Training readiness: "+str(readiness["training_ready"]), flush=True)


if __name__ == "__main__":
    main()
