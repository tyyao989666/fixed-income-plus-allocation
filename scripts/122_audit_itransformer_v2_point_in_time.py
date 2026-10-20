from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import load_daily_config  # noqa: E402
from src.itransformer_daily_timing_v2 import v2_path  # noqa: E402
from src.v1_strategy import load_config  # noqa: E402


def main() -> None:
    daily_cfg = load_daily_config()
    strategy_cfg = load_config()
    frame = pd.read_parquet(v2_path(daily_cfg, "timing_v2_frame"))
    monthly = pd.read_parquet(v2_path(daily_cfg, "timing_v2_monthly_oos"))
    gate = json.loads(v2_path(daily_cfg, "timing_v2_gate_report").read_text(encoding="utf-8"))
    cv = pd.read_csv(v2_path(daily_cfg, "timing_v2_cv_report"))
    overlay = strategy_cfg["signal_overlays"]["itransformer_rank"]

    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"], errors="coerce")
    monthly["as_of_date"] = pd.to_datetime(monthly["as_of_date"], errors="coerce")
    monthly["training_cutoff"] = pd.to_datetime(monthly["training_cutoff"], errors="coerce")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": str(evidence)})

    labelled = frame[frame["has_label"].astype(bool)]
    add("labels_strictly_after_decision", bool((labelled["label_end_date"] > labelled["as_of_date"]).all()), len(labelled))
    add("monthly_training_cutoff_before_decision", bool((monthly["training_cutoff"] < monthly["as_of_date"]).all()), len(monthly))
    counts = monthly.groupby("as_of_date")["sleeve"].nunique()
    add("five_sleeves_per_month", bool(counts.eq(5).all()), counts.value_counts().to_dict())
    add("no_duplicate_month_sleeve", not monthly.duplicated(["as_of_date", "sleeve"]).any(), len(monthly))
    overlay_enabled = bool(overlay.get("enabled", False))
    gate_open = bool(gate.get("enabled_for_formal_strategy", False))
    add(
        "formal_gate_consistent",
        overlay_enabled == gate_open,
        {"overlay_enabled": overlay_enabled, "gate_open": gate_open},
    )
    add(
        "shadow_disabled_when_gate_closed",
        gate_open or not overlay_enabled,
        {"overlay_enabled": overlay_enabled, "gate_open": gate_open},
    )
    model_cv = cv[cv["source"].eq("identified_itransformer") & cv["fold"].str.startswith("wf_")]
    add(
        "walkforward_results_present",
        len(model_cv) == 3 and model_cv["rank_ic"].notna().all(),
        model_cv[["fold", "rank_ic"]].to_dict(orient="records"),
    )
    add("balanced_only", overlay.get("enabled_profiles") == ["balanced"], overlay.get("enabled_profiles"))
    add("direct_single_channel", bool(overlay.get("direct_tilt_enabled")) and not bool(overlay.get("ep_mean_enabled")), {"direct": overlay.get("direct_tilt_enabled"), "ep": overlay.get("ep_mean_enabled")})
    add("tilt_cap_at_1_5pct", float(overlay.get("tilt", 1.0)) <= 0.015, overlay.get("tilt"))
    add("live_gate_required", bool(overlay.get("require_live_gate")), overlay.get("gate_report_path"))

    report = pd.DataFrame(checks)
    csv_path = ROOT / "output" / "reports" / "itransformer_v2_point_in_time_audit.csv"
    md_path = ROOT / "output" / "reports" / "itransformer_v2_point_in_time_audit.md"
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = ["# iTransformer V2 Point-in-Time 审计", "", f"- 检查数：{len(report)}", f"- 失败数：{int((~report['passed']).sum())}", "", report.to_markdown(index=False)]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path.relative_to(ROOT)), "md": str(md_path.relative_to(ROOT)), "checks": len(report), "failed": int((~report["passed"]).sum())}, ensure_ascii=False))
    if not report["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
