from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "reports"


def parse_profiles(value: str) -> list[str]:
    return ["low_vol", "balanced", "high_vol"] if value == "all" else [x.strip() for x in value.split(",") if x.strip()]


def read_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return frame.set_index("metric")["value"].astype(float).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic and agentic V1 outputs.")
    parser.add_argument("--profiles", default="all")
    args = parser.parse_args()
    rows = []
    for profile in parse_profiles(args.profiles):
        baseline_path = REPORTS / f"v1_{profile}_backtest_metrics.csv"
        agentic_path = REPORTS / f"v1_agentic_{profile}_backtest_metrics.csv"
        if not baseline_path.exists() or not agentic_path.exists():
            continue
        baseline = read_metrics(baseline_path)
        agentic = read_metrics(agentic_path)
        for version, values in [("no_agent", baseline), ("agentic", agentic)]:
            rows.append({"profile": profile, "version": version, **values})
    out = pd.DataFrame(rows)
    audit_path = REPORTS / "agentic_market_view_audit.jsonl"
    audit_rows = []
    if audit_path.exists():
        audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_fallback = bool(audit_rows) and all(bool(row.get("fallback")) for row in audit_rows)
    if not out.empty:
        out["agentic_effective"] = not all_fallback
    csv_path = REPORTS / "v1_agentic_comparison.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = ["# V1 No-Agent vs Agentic Comparison", ""]
    if out.empty:
        lines.append("Comparison unavailable: one or more version outputs are missing.")
    else:
        cols = [x for x in ["profile", "version", "annual_return", "annual_vol", "sharpe", "max_drawdown", "avg_turnover"] if x in out.columns]
        lines.append(out[cols].to_markdown(index=False))
        if all_fallback:
            lines.extend([
                "",
                "> 当前所有 Agent 日期均因模型不可用而回退机械 views；这不是有效的 Agent 绩效比较。",
                "> 两组指标若有差异，只说明正式文件和本次运行所用代码/配置快照不同。",
            ])
        lines.extend(["", "Agentic results are experimental and do not overwrite the formal no-agent files."])
    (REPORTS / "v1_agentic_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"comparison rows={len(out)}; output={csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
