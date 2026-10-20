from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_ppo_hybrid import (  # noqa: E402
    audit_hybrid,
    load_hybrid_config,
    run_profile,
    write_comparison,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the point-in-time iTransformer V2 + PPO hybrid pipeline.")
    parser.add_argument("--profiles", default="low_vol,balanced,high_vol")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_hybrid_config(Path(args.config) if args.config else None)
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    results = []
    for index, profile in enumerate(profiles, 1):
        print(f"[{index}/{len(profiles)}] build/train/backtest {profile}", flush=True)
        result = run_profile(profile, cfg)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary, yearly, gate = write_comparison(cfg)
    audit = audit_hybrid(cfg)
    write_report(summary, yearly, gate, cfg)
    payload = {
        "profiles": results,
        "summary": str((ROOT / cfg["outputs"]["summary"]).relative_to(ROOT)),
        "gate": str((ROOT / cfg["outputs"]["gate_report"]).relative_to(ROOT)),
        "audit_failed": int((~audit["passed"]).sum()),
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if not audit["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
