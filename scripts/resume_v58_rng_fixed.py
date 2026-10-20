"""Run the unchanged V58 pipeline with device-correct checkpoint RNG restoration."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import subprocess
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cuda_rng_resume import cuda_rng_restore
from src.cuda_finite_check import noncompacting_finite_check


@contextmanager
def corrected_children():
    original = subprocess.run

    def run(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) > 4:
            path = Path(str(command[4])).resolve()
            allowed = [ROOT / "scripts/train_v57_multitask.py", ROOT / "scripts/gate_v57_forecasts.py"]
            if path in allowed:
                command = list(command[:4]) + [str(Path(__file__)), "--child", str(path)] + list(command[5:])
        return original(command, *args, **kwargs)

    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original


if __name__ == "__main__":
    child = len(sys.argv) > 2 and sys.argv[1] == "--child"
    entry = Path(sys.argv[2]).resolve() if child else ROOT / "scripts/run_v58_rebuild_pipeline.py"
    if child:
        if entry not in [ROOT / "scripts/train_v57_multitask.py", ROOT / "scripts/gate_v57_forecasts.py"]:
            raise ValueError("Unapproved child entry")
        sys.argv = [str(entry)] + sys.argv[3:]
    provenance = {
        "pid": os.getpid(),
        "entry": str(entry),
        "scope": "CPU uint8 RNG restore and equivalent noncompacting finite check; no reseeding or weight/config/data changes",
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "runtime_sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "src/cuda_rng_resume.py", ROOT / "src/cuda_finite_check.py"]
        },
    }
    target = ROOT / "output/reports/corporate_action_v58/rng_resume_provenance.json"
    target.with_name(f"runtime_provenance_{os.getpid()}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if not child:
        target.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    with cuda_rng_restore(), noncompacting_finite_check(), corrected_children():
        runpy.run_path(str(entry), run_name="__main__")
