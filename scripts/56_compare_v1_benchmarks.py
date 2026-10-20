from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v1_full_pipeline import PROFILE_OUTPUTS, parse_profiles, profile_cfg, run_profile  # noqa: E402
from src.v1_strategy import RISK_SLEEVES  # noqa: E402


PROFILES = ["low_vol", "balanced", "high_vol"]


def output_template(variant: str) -> dict[str, str]:
    out = {}
    for key, value in PROFILE_OUTPUTS.items():
        path = Path(value)
        out[key] = str(path.with_name(f"{path.stem}_{variant}{path.suffix}")).replace("\\", "/")
    return out


def metrics_as_dict(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def variant_cfg(profile: str, variant: str) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    cfg["outputs"] = {k: v.format(profile=profile) for k, v in output_template(variant).items()}

    if variant == "enhanced":
        return cfg

    if variant == "risk_budget_only":
        cfg["view_tilts"] = {k: 0.0 for k in cfg.get("view_tilts", {})}
        for overlay in cfg.setdefault("signal_overlays", {}).values():
            overlay["enabled"] = False
        return cfg

    if variant == "equal_risk_parity":
        cfg["view_tilts"] = {k: 0.0 for k in cfg.get("view_tilts", {})}
        for overlay in cfg.setdefault("signal_overlays", {}).values():
            overlay["enabled"] = False
        equal = 1.0 / len(RISK_SLEEVES)
        cfg["product"]["profiles"][profile]["target_risk_contribution"] = {s: equal for s in RISK_SLEEVES}
        return cfg

    raise ValueError(f"Unknown benchmark variant: {variant}")


def run_variant(profile: str, variant: str) -> dict[str, Any]:
    original = profile_cfg(profile)
    cfg = variant_cfg(profile, variant)

    import scripts.run_v1_full_pipeline as runner

    old_profile_cfg = runner.profile_cfg
    try:
        runner.profile_cfg = lambda _: copy.deepcopy(cfg)
        result = run_profile(profile)
    finally:
        runner.profile_cfg = old_profile_cfg

    metrics = metrics_as_dict(ROOT / cfg["outputs"]["metrics"])
    result.update(
        {
            "variant": variant,
            "annual_return": metrics.get("annual_return"),
            "annual_vol": metrics.get("annual_vol"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "calmar": metrics.get("calmar"),
            "avg_turnover": metrics.get("avg_turnover"),
        }
    )
    return result


def main() -> None:
    variants = ["enhanced", "risk_budget_only", "equal_risk_parity"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for profile in PROFILES:
            print(f"=== {variant} / {profile} ===", flush=True)
            rows.append(run_variant(profile, variant))

    report = pd.DataFrame(rows)
    path = ROOT / "output" / "reports" / "v1_benchmark_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "variant",
        "profile",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "avg_turnover",
        "backtest_months",
    ]
    report[cols].to_csv(path, index=False, encoding="utf-8-sig")

    print(json.dumps({"report": str(path.relative_to(ROOT)), "rows": len(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
