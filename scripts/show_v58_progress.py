"""Compact actual-process and on-disk progress for corrected-data rebuilding."""
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_v45_research_pipeline import process_alive


def main():
    report=ROOT/"output/reports/corporate_action_v58"
    launch=json.loads((report/"pipeline_launch.json").read_text(encoding="utf-8-sig"))
    state=json.loads((report/"pipeline_status.json").read_text())
    base=ROOT/"output/models/corporate_action_v58"
    count=sum(len(json.loads(p.read_text()).get("completed_seeds",[])) for p in base.glob("*/progress.json"))
    head=ROOT/"output/models/multitask_v58"
    head_count=len(list(head.glob("*/*/summary.json")))
    logs=list(base.glob("*/seed_*.csv")) if state["stage"]=="fresh_iTransformer" else list(head.glob("*/*/loss.csv"))
    last=None
    if logs:
        latest=max(logs,key=lambda p:p.stat().st_mtime_ns)
        with latest.open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
        if rows:last={k:rows[-1].get(k) for k in ["year","seed","stage","epoch","selected_epoch","best_epoch","train_loss","validation_loss"]}
    alive=process_alive(state["pid"])
    effective=state["status"]
    if not alive and effective.startswith("training"):
        effective="interrupted_no_live_process"
    print(json.dumps(dict(status=effective,recorded_status=state["status"],stage=state["stage"],pid=state["pid"],
        process_alive=alive,fresh_encoder_seed_tasks=f"{count}/15",
        distribution_head_tasks=f"{head_count}/15",last_epoch=last,
        stdout=launch["stdout"],stderr=launch["stderr"],formal_promotion=False),indent=2))


if __name__=="__main__":main()
