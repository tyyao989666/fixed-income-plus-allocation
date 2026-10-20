"""Print the current v45 stage and latest real epoch; optionally watch."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def last_record(path):
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size-32768))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, TypeError):
            continue
    return {}


def snapshot():
    manifest = ROOT / "output/reports/v45_pipeline_manifest.json"
    state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    logs = list((ROOT / "output/models/v45_forecast").glob("*/*.jsonl"))
    logs += list((ROOT / "output/reports").glob("v45_*_ppo_training.jsonl"))
    latest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
    row = last_record(latest) if latest else {}
    return dict(at=datetime.now().isoformat(timespec="seconds"), status=state.get("status", "not_started"),
        stage=state.get("current_stage"), pid=state.get("child_pid"),
        completed_stages=[r["stage"] for r in state.get("stages", []) if r.get("status") == "completed"],
        latest_epoch=row, epoch_log=str(latest) if latest else None,
        error=state.get("error"), checkpoint_file_count=len(list((ROOT / "output/models/v45_forecast").glob("*/*.pt"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be >=1 second")
    while True:
        state = snapshot()
        print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)
        if not args.watch or state["status"] in ("failed", "interrupted", "completed"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
