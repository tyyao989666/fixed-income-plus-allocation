from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    config = "config/itransformer_daily_ohlcv_config.yaml"
    run("build OHLCV daily-path dataset", [PYTHON, "scripts/65_build_itransformer_daily_dataset.py", "--config", config])
    run("train OHLCV multi-seed ensemble", [PYTHON, "scripts/70_run_itransformer_daily_enhanced.py", "--config", config, "--skip-build"])
    run("blend original and OHLCV signals", [PYTHON, "scripts/71_blend_itransformer_daily_signals.py"])
    run("walk-forward validation", [PYTHON, "scripts/73_walkforward_itransformer_daily.py", "--config", config])
    run("compare portfolio impact", [PYTHON, "scripts/72_compare_blended_itransformer_portfolios.py"])
    print("\nEnhancement pipeline complete. Formal V1 signal path is unchanged pending acceptance.", flush=True)


if __name__ == "__main__":
    main()
