"""Read actual V56 launch identity and latest epoch; never starts a process."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_v45_research_pipeline import process_alive


def main():
    root = ROOT / "output/reports/total_view_rank_v56"
    launch = json.loads((root / "GPU_training_launch.json").read_text(encoding="utf-8-sig"))
    path = root / "training_status.json"
    state = json.loads(path.read_text()) if path.exists() else {"status": "preflight_or_initializing"}
    model = ROOT / "output/models/total_view_rank_v56"
    complete = 0
    for p in model.glob("*/progress.json"):
        complete += len(json.loads(p.read_text()).get("completed_seeds", []))
    logs = list(model.glob("*/seed_*.csv"))
    last = None
    if logs:
        latest = max(logs, key=lambda p: p.stat().st_mtime_ns)
        with latest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            last = {k: rows[-1][k] for k in ["year", "seed", "stage", "epoch", "selected_epoch", "train_loss", "validation_loss"]}
    alive = process_alive(launch["pid"])
    print(json.dumps(dict(status=state["status"], process_alive=alive, pid=launch["pid"],
        completed_annual_seeds=complete, expected=15, last_epoch=last,
        stale_training_flag=not alive and state["status"] == "training",
        stdout=launch["stdout"], stderr=launch["stderr"], formal_promotion=False), indent=2))


if __name__ == "__main__":
    main()
