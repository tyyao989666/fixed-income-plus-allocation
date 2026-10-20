from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.integrated_daily_view_ppo import (  # noqa: E402
    audit_and_report,
    build_state_frame,
    load_config,
    run_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the nine-grid-first integrated iTransformer-view + PPO + EP/skfolio pipeline."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profiles", default="low_vol,balanced,high_vol")
    args = parser.parse_args()
    cfg = load_config(args.config)
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    cfg["profiles"] = profiles
    state = build_state_frame(cfg)
    print(
        json.dumps(
            {
                "stage": "state_ready",
                "daily_rows": len(state),
                "state_features": len([column for column in state if column.startswith("state_")]),
                "start": str(state.date.min().date()),
                "end": str(state.date.max().date()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = []
    for index, profile in enumerate(profiles, 1):
        print(f"[{index}/{len(profiles)}] integrated view PPO {profile}", flush=True)
        result = run_profile(profile, state, cfg)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary, audit, gate = audit_and_report(cfg)
    print(
        json.dumps(
            {
                "results": results,
                "audit_failed": int((~audit.passed).sum()),
                "formal_gate": gate["enabled_for_formal_strategy"],
                "summary_rows": len(summary),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not audit["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
