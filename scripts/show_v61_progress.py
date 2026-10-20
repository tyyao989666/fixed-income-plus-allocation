"""Read-only actual process status for the conditionally supervised ranker."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_v45_research_pipeline import process_alive

out = ROOT / "output/reports/state_momentum_v61"
state = json.loads((out / "progress.json").read_text())
alive = process_alive(state["pid"])
status = state["status"]
if status == "training" and not alive:
    status = "interrupted_no_live_process"
print(json.dumps(dict(status=status, recorded_status=state["status"], process_alive=alive,
    pid=state["pid"], completed=state["completed"], total=state["total"], current=state.get("current"),
    fresh_full_encoder=state["fresh_full_encoder"], value_boundary_enabled=False,
    decision_available=(out / "decision.json").exists(), formal_promotion=False, ppo_training_started=False), indent=2))
