from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


def main() -> None:
    errors: list[str] = []
    agent_ep_path = PROCESSED / "ep_views_agentic_daily_v1.parquet"
    base_ep_path = PROCESSED / "ep_views_daily_v1.parquet"
    audit_path = REPORTS / "agentic_market_view_audit.jsonl"
    for path in [agent_ep_path, audit_path]:
        if not path.exists():
            errors.append(f"missing:{path.relative_to(ROOT)}")

    audits = []
    if audit_path.exists():
        audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_fallback = bool(audits) and all(bool(row.get("fallback")) for row in audits)
    if all_fallback and base_ep_path.exists() and agent_ep_path.exists():
        base = pd.read_parquet(base_ep_path)
        agent = pd.read_parquet(agent_ep_path)
        columns = sorted(set(base.columns).intersection(agent.columns))
        base = base[columns].sort_values(columns).reset_index(drop=True)
        agent = agent[columns].sort_values(columns).reset_index(drop=True)
        if not base.equals(agent):
            errors.append("all-fallback agent EP views differ from mechanical EP views")

    checked_profiles = 0
    for profile in ["low_vol", "balanced", "high_vol"]:
        path = PROCESSED / f"v1_agentic_{profile}_target_weights_monthly.parquet"
        if not path.exists():
            continue
        checked_profiles += 1
        weights = pd.read_parquet(path)
        sums = weights.groupby("date")["weight"].sum()
        if not np.allclose(sums.to_numpy(), 1.0, atol=1e-6):
            errors.append(f"{profile}:weights_do_not_sum_to_one")
        cash = weights.loc[weights["asset_code"].astype(str).eq("cash"), "weight"]
        if not cash.empty and float(cash.abs().max()) > 1e-12:
            errors.append(f"{profile}:cash_not_zero")

    result = {
        "status": "failed" if errors else "ok",
        "audit_dates": len(audits),
        "all_fallback": all_fallback,
        "checked_profiles": checked_profiles,
        "errors": errors,
    }
    (REPORTS / "agentic_pipeline_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

