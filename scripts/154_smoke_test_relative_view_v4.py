from __future__ import annotations

"""Fast structural test for the relative-view target and execution controller."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robust_execution_ppo import (  # noqa: E402
    RobustExecutionEnvironment,
    build_state_frame,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "view_execution_ppo_v4_config.yaml",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional strict-OOS prediction artifact used only for this test.",
    )
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()

    cfg, base_cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    base_cfg = copy.deepcopy(base_cfg)
    if args.predictions is not None:
        prediction_path = (
            args.predictions
            if args.predictions.is_absolute()
            else ROOT / args.predictions
        )
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        relative = str(prediction_path.relative_to(ROOT))
        cfg["data"]["predictions"] = relative
        base_cfg["data"]["predictions"] = relative
        base_cfg["actions"]["require_formal_itransformer_gate"] = False
    state = build_state_frame(cfg, base_cfg, write_output=False)
    if state.empty:
        raise AssertionError("state frame is empty")

    available = pd.to_numeric(
        state["state_it_available"], errors="coerce"
    ).fillna(0.0).gt(0.5)
    cutoff = pd.to_datetime(
        state["itransformer_information_cutoff"], errors="coerce"
    )
    dates = pd.to_datetime(state["date"], errors="coerce")
    if not bool((~available | (cutoff.notna() & cutoff.le(dates))).all()):
        raise AssertionError("iTransformer information cutoff exceeds decision date")

    reports: list[dict] = []
    for profile in cfg["profiles"]:
        env = RobustExecutionEnvironment(profile, state, cfg, base_cfg)
        sample = np.unique(
            np.linspace(
                0,
                len(env.state) - 2,
                min(int(args.samples), len(env.state) - 1),
                dtype=int,
            )
        )
        active_targets = 0
        qualified_horizons = 0
        max_sum_error = 0.0
        max_bound_error = 0.0
        representative = int(sample[-1])
        for index in sample:
            target, diagnostics = env.full_target(int(index))
            max_sum_error = max(max_sum_error, abs(float(target.sum()) - 1.0))
            max_bound_error = max(
                max_bound_error,
                float(np.maximum(env.base.lo - target, 0.0).max()),
                float(np.maximum(target - env.base.hi, 0.0).max()),
            )
            active_targets += int(diagnostics["itransformer_active"])
            qualified_horizons += int(
                diagnostics["itransformer_horizon_count"]
            )
            for horizon in base_cfg["actions"]["horizons_days"]:
                if diagnostics[f"it_h{horizon}_qualified"] > 0.5:
                    matured = diagnostics[
                        f"it_h{horizon}_matured_observations"
                    ]
                    minimum = int(
                        base_cfg["data"]["reliability_min_observations"]
                    )
                    if matured < minimum:
                        raise AssertionError(
                            f"{profile} h{horizon} qualified with only "
                            f"{matured} matured observations"
                        )
            if diagnostics["itransformer_active"]:
                representative = int(index)
        if max_sum_error > 1e-8 or max_bound_error > 1e-8:
            raise AssertionError(
                f"{profile} target constraint error: "
                f"sum={max_sum_error}, bounds={max_bound_error}"
            )
        if active_targets <= 0 or qualified_horizons <= 0:
            raise AssertionError(
                f"{profile} never activated the calibrated iTransformer residual view"
            )

        current = env.initial_weights(representative)
        action_rows: list[dict] = []
        previous_turnover = -1.0
        for action, configured_fraction in enumerate(env.fractions):
            _, _, _, _, info = env.step(
                representative,
                action,
                current,
                env.initial_context(),
            )
            if not np.isclose(
                float(info["action_fraction"]),
                float(configured_fraction),
                atol=1e-12,
            ):
                raise AssertionError("execution fraction is not a configured action")
            if not pd.Timestamp(info["execution_start"]) > pd.Timestamp(
                env.state.loc[representative, "date"]
            ):
                raise AssertionError("execution is not T+1")
            if float(info["turnover"]) + 1e-10 < previous_turnover:
                raise AssertionError("turnover is not monotone across progress actions")
            previous_turnover = float(info["turnover"])
            action_rows.append(
                {
                    "fraction": float(info["action_fraction"]),
                    "turnover": float(info["turnover"]),
                    "linear_cost": float(info["linear_cost"]),
                    "impact_cost": float(info["impact_cost"]),
                }
            )
        reports.append(
            {
                "profile": profile,
                "sampled_targets": len(sample),
                "active_targets": active_targets,
                "qualified_horizons": qualified_horizons,
                "max_weight_sum_error": max_sum_error,
                "max_bound_error": max_bound_error,
                "actions": action_rows,
            }
        )

    print(
        json.dumps(
            {
                "passed": True,
                "state_rows": len(state),
                "state_start": str(pd.Timestamp(state["date"].min()).date()),
                "state_end": str(pd.Timestamp(state["date"].max()).date()),
                "profiles": reports,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
