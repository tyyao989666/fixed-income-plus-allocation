from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

sys.path.insert(0, str(ROOT))

from src.itransformer_signal import load_config, out_path  # noqa: E402


def suffix_for(seed: int) -> str:
    return f"_seed{seed}"


def run_seed(seed: int) -> None:
    subprocess.run(
        [PYTHON, "scripts/09_train_itransformer.py", "--seed", str(seed), "--output-suffix", suffix_for(seed)],
        cwd=ROOT,
        check=True,
    )


def suffixed_output(cfg: dict, key: str, seed: int) -> Path:
    path = Path(cfg["outputs"][key])
    return ROOT / path.with_name(f"{path.stem}{suffix_for(seed)}{path.suffix}")


def main() -> None:
    cfg = load_config()
    seeds = [int(x) for x in cfg.get("experiment", {}).get("seeds", [int(cfg["model"]["seed"])])]
    reports = []
    for seed in seeds:
        print(f"\n=== train iTransformer seed {seed} ===", flush=True)
        run_seed(seed)
        report_path = suffixed_output(cfg, "validation_report", seed)
        report = pd.read_csv(report_path)
        report["seed"] = seed
        reports.append(report)

    all_report = pd.concat(reports, ignore_index=True)
    all_path = out_path(cfg, "multiseed_report")
    all_path.parent.mkdir(parents=True, exist_ok=True)
    all_report.to_csv(all_path, index=False, encoding="utf-8-sig")

    metric_cols = ["mse", "rank_ic", "top1_hit_rate", "top2_realized_return", "long_short_spread"]
    summary = (
        all_report.groupby(["model", "split"], as_index=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns.to_flat_index()]
    summary_path = out_path(cfg, "multiseed_summary")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "seeds": seeds,
                "report": cfg["outputs"]["multiseed_report"],
                "summary": cfg["outputs"]["multiseed_summary"],
                "rows": len(all_report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
