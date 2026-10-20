from __future__ import annotations

"""Single entry point for the frozen mechanical/no-Agent release.

It intentionally does not call any LLM and does not retrain iTransformer or
PPO by default. Expensive research components are opt-in flags so a routine
release refresh cannot accidentally overwrite the audited baseline.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the current no-Agent research release.")
    parser.add_argument("--refresh-formal-monthly", action="store_true")
    parser.add_argument("--replay-robust", action="store_true")
    parser.add_argument("--train-robust", action="store_true")
    args = parser.parse_args()
    if args.replay_robust and args.train_robust:
        parser.error("choose at most one of --replay-robust and --train-robust")

    _run("build point-in-time market state cards", ["scripts/10_generate_market_state_cards.py"])
    _run(
        "build deterministic mechanical views",
        ["scripts/11_run_llm_scenario_engine.py", "--mode", "mechanical", "--active-source", "mechanical"],
    )
    _run("build EP/skfolio input views", ["scripts/12_build_ep_views.py", "--source", "mechanical"])

    if args.refresh_formal_monthly:
        _run(
            "run monthly V1 formal baseline",
            [
                "scripts/run_v1_full_pipeline.py", "--skip-data", "--skip-market-skill",
                "--skip-itransformer-ppo", "--strategy-version", "no_agent",
                "--profiles", "low_vol,balanced,high_vol", "--workers", "3",
            ],
        )
    if args.replay_robust:
        _run("replay Robust PPO", ["scripts/149_run_robust_execution_ppo.py", "--skip-training"])
    if args.train_robust:
        _run("train Robust PPO", ["scripts/149_run_robust_execution_ppo.py"])

    _run("audit current release", ["scripts/150_audit_current_release.py"])
    print("\nRelease refresh complete. See output/reports/current_release_audit.md", flush=True)


if __name__ == "__main__":
    main()
