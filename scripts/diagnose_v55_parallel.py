"""Summarize completed frozen annual fits without changing training or gates."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "output/reports/temporal_opportunity_v55"
    rows, failures = [], []
    roots = {
        "V53_raw_return": ROOT / "output/models/unified_v53/full",
        "V55_opportunity": ROOT / "output/models/temporal_opportunity_v55",
    }
    # Discover the registered CPU output rather than substituting archived pilots.
    for arm, root in roots.items():
        for path in sorted(root.glob("*/report.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            gates = d.get("gates", [])
            counts = Counter(
                name for gate in gates
                for name, passed in gate.get("checks", {}).items() if not passed
            )
            rows.append({
                "arm": arm, "year": d["year"],
                "selected_candidate": d.get("selected_candidate"),
                "completed_seeds": len(d.get("completed_seeds", [])),
                "gate_cells": len(gates),
                "active_cells": sum(bool(g.get("active")) for g in gates),
                "predictive_evidence_pass": d.get("predictive_evidence_pass", False),
                "portfolio_stage_allowed": d.get("portfolio_stage_allowed", False),
                "source": str(path.relative_to(ROOT)),
            })
            failures.extend({"arm": arm, "year": d["year"], "check": k,
                             "failed_cells": v, "total_cells": len(gates)}
                            for k, v in counts.items())
    pd.DataFrame(rows).to_csv(out / "parallel_annual_diagnostics.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "parallel_gate_failure_counts.csv", index=False)
    note = """# V53 / V55 Parallel Diagnostics

Only completed annual report artifacts are summarized. Missing years remain pending.
The two targets have different units and warm-up samples: do not compare loss levels
or selection-score magnitudes across arms. Compare decoded return predictions on a
common observed-date/entity/horizon mask before claiming an improvement.

Gate failure frequencies identify research bottlenecks, not permission to relax gates.
2024-2026 have already been inspected in prior research and are not a new untouched
holdout. Any further objective/model selection must use historical training/validation
only, preserve the nine-grid and 38 entities, and record all attempted variants.
No formal portfolio promotion or PPO reward certification is implied by this table.
"""
    (out / "parallel_diagnostics.md").write_text(note, encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
