"""Read-only CPU control / CUDA challenger progress from actual process and logs."""
from pathlib import Path
import csv
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_v45_research_pipeline import process_alive


def main():
    arms=[("CPU_raw_relative_control",ROOT/"output/reports/unified_v53/full_training_launch.json",
        ROOT/"output/models/unified_v53/full"),
        ("GPU_temporal_opportunity",ROOT/"output/reports/temporal_opportunity_v55/GPU_training_launch.json",
        ROOT/"output/models/temporal_opportunity_v55")]
    for name,launch_path,model_dir in arms:
        if not launch_path.exists():continue
        launch=json.loads(launch_path.read_text(encoding="utf-8-sig"))
        complete=[]
        for p in sorted(model_dir.glob("*/progress.json")):
            f=json.loads(p.read_text(encoding="utf-8"))
            complete.extend((p.parent.name,seed) for seed in f.get("completed_seeds",[]))
        logs=list(model_dir.glob("*/seed_*.csv"))
        last=None
        if logs:
            latest=max(logs,key=lambda p:p.stat().st_mtime_ns)
            with latest.open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
            if rows:
                row=rows[-1]
                last=dict(year=row["year"],seed=row["seed"],stage=row["stage"],epoch=row["epoch"],
                    selected_epoch=row["selected_epoch"],train_loss=row["train_loss"],validation_loss=row["validation_loss"])
        print(json.dumps(dict(arm=name,pid=launch["pid"],process_alive=process_alive(launch["pid"]),
            completed_annual_seeds=len(complete),expected_annual_seeds=15,last_epoch=last,
            stdout=launch["stdout"],formal_promotion=False),ensure_ascii=False))


if __name__=="__main__":main()
