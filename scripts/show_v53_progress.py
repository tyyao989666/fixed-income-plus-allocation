"""Read-only real training progress; no model reload and no network requests."""
from pathlib import Path
import csv
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_v45_research_pipeline import process_alive


def main():
    for name in ["pilot_forecast", "forecast"]:
        status_path = ROOT / "output/reports/unified_v53" / name / "training_status.json"
        if status_path.exists():
            state = json.loads(status_path.read_text(encoding="utf-8"))
            effective = state["status"]
            launch_path = ROOT / "output/reports/unified_v53/full_training_launch.json"
            live = None
            if name == "forecast" and launch_path.exists():
                live = process_alive(json.loads(launch_path.read_text(encoding="utf-8-sig"))["pid"])
                if effective in ["training", "preflight"] and not live:
                    effective = "interrupted_process_missing_NOT_running"
            print(name, effective, "process_alive=", live, "formal_promotion=", state["formal_promotion"])
    for path in sorted((ROOT / "output/models/unified_v53").rglob("seed_*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        latest = rows[-1]
        elapsed = sum(float(row["seconds"]) for row in rows)
        values = [float(row["train_loss"]) for row in rows]
        evaluated = [r for r in rows if r["evaluated"] == "True"]
        finite = all(math.isfinite(v) for v in values)
        print(json.dumps(dict(log=str(path.relative_to(ROOT)), epoch=int(latest["epoch"]),
            selected_epoch=int(latest["selected_epoch"]), train_loss=float(latest["train_loss"]),
            first_train_loss=values[0], validation_loss=float(latest["validation_loss"]),
            evaluated_points=len(evaluated), finite_training_losses=finite,
            training_minutes=round(elapsed/60,2),
            note="pretraining validation_score is not applicable; inspect validation_loss" if "pretraining" in path.name else "selection uses validation score, not smallest training loss"), ensure_ascii=False))


if __name__ == "__main__":
    main()
