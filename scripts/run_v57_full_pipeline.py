"""One entry point: triage -> distributional transfer -> unchanged gates -> stop safely."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    for script in ["test_v57_multitask.py","audit_v57_rebuild.py","train_v57_multitask.py","gate_v57_forecasts.py"]:
        print("STAGE "+script,flush=True)
        subprocess.run([sys.executable,"-u","-X","utf8",str(ROOT/"scripts"/script)],cwd=ROOT,check=True)
    out=ROOT/"output/reports/multitask_v57"
    readiness=json.loads((out/"execution_readiness.json").read_text())
    print(json.dumps(dict(status="research_complete_execution_blocked",readiness=readiness),indent=2),flush=True)
    # Never train PPO on the old unverified returns merely because this process finished.
    return 0


if __name__=="__main__":sys.exit(main())
