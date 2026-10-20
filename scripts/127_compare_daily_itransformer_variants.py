from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "reports"

VARIANTS = {
    "baseline_32d_1l": {
        "cv": REPORTS / "itransformer_daily_timing_v2_cv_report.csv",
        "log": REPORTS / "itransformer_daily_timing_v2_training_log.csv",
        "gate": REPORTS / "itransformer_daily_timing_v2_formal_gate.json",
    },
    "direct_64d_2l": {
        "cv": REPORTS / "itransformer_daily_timing_v2_challenger_cv_report.csv",
        "log": REPORTS / "itransformer_daily_timing_v2_challenger_training_log.csv",
        "gate": REPORTS / "itransformer_daily_timing_v2_challenger_formal_gate.json",
    },
    "residual_64d_2l": {
        "cv": REPORTS / "itransformer_daily_timing_v2_residual_cv_report.csv",
        "log": REPORTS / "itransformer_daily_timing_v2_residual_training_log.csv",
        "gate": REPORTS / "itransformer_daily_timing_v2_residual_formal_gate.json",
    },
}


def main() -> None:
    cv_parts: list[pd.DataFrame] = []
    gate_rows: list[dict] = []
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    colors = {"baseline_32d_1l": "#5B8FF9", "direct_64d_2l": "#F6BD16", "residual_64d_2l": "#5AD8A6"}
    for variant, paths in VARIANTS.items():
        if paths["cv"].exists():
            cv = pd.read_csv(paths["cv"])
            cv.insert(0, "variant", variant)
            cv_parts.append(cv)
        if paths["gate"].exists():
            gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
            gate_rows.append(
                {
                    "variant": variant,
                    "formal_gate": bool(gate.get("enabled_for_formal_strategy", False)),
                    "cv_median_rank_ic": gate.get("cv_median_rank_ic"),
                    "cv_min_rank_ic": gate.get("cv_min_rank_ic"),
                    "baseline_cv_median_rank_ic": gate.get("baseline_cv_median_rank_ic"),
                    "incremental_cv_rank_ic": gate.get("incremental_cv_rank_ic"),
                    "holdout_rank_ic": gate.get("holdout_rank_ic"),
                    "fixed_epochs": gate.get("fixed_epochs"),
                    "final_residual_alpha": gate.get("final_residual_alpha"),
                    "fold_residual_alphas": json.dumps(gate.get("fold_residual_alphas", [])),
                }
            )
        if paths["log"].exists():
            log = pd.read_csv(paths["log"])
            curve = log.groupby("epoch", as_index=False).agg(train_loss=("train_loss", "mean"), val_rank_ic=("val_rank_ic", "mean"))
            axes[0].plot(curve["epoch"], curve["train_loss"], label=variant, color=colors[variant], linewidth=1.8)
            axes[1].plot(curve["epoch"], curve["val_rank_ic"], label=variant, color=colors[variant], linewidth=1.8)
    cv_all = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
    gates = pd.DataFrame(gate_rows)
    cv_path = REPORTS / "itransformer_daily_model_variant_cv_comparison.csv"
    gate_path = REPORTS / "itransformer_daily_model_variant_gate_comparison.csv"
    cv_all.to_csv(cv_path, index=False, encoding="utf-8-sig")
    gates.to_csv(gate_path, index=False, encoding="utf-8-sig")
    axes[0].set_title("Training loss by epoch (fold/seed mean)")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Validation RankIC by epoch (fold/seed mean)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RankIC")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure_path = REPORTS / "itransformer_daily_model_training_curves.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    model_cv = cv_all[cv_all.get("source", pd.Series(dtype=str)).eq("identified_itransformer")].copy()
    lines = [
        "# Daily iTransformer Variant Comparison",
        "",
        "## Formal Gate",
        "",
        gates.to_markdown(index=False) if not gates.empty else "No gate files.",
        "",
        "## Walk-Forward RankIC",
        "",
        model_cv[["variant", "fold", "rank_ic", "top_bottom_raw_spread"]].to_markdown(index=False)
        if not model_cv.empty else "No CV files.",
        "",
        "The training curve is an optimization diagnostic. A falling training loss does not by itself establish out-of-sample alpha.",
    ]
    (REPORTS / "itransformer_daily_model_variant_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cv": str(cv_path.relative_to(ROOT)), "gates": str(gate_path.relative_to(ROOT)), "figure": str(figure_path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()

