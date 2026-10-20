from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daily_rl_dataset import build_daily_sleeve_return_matrix, load_overlay_config  # noqa: E402
from src.daily_timing_cards import build_daily_timing_cards, load_timing_card_config  # noqa: E402
from src.itransformer_signal import RISK_SLEEVES  # noqa: E402
from src.ppo_overlay_env import project_simplex_bounds  # noqa: E402


def bounds_for(profile: str, overlay_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = overlay_cfg["action_space"]["sleeve_bounds"][profile]
    lo = np.array([bounds[s][0] for s in RISK_SLEEVES], dtype=float)
    hi = np.array([bounds[s][1] for s in RISK_SLEEVES], dtype=float)
    return lo, hi


def tilt_weights(row: pd.Series, base: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if not bool(row["allow_model_timing"]) or str(row["action_gate"]) == "no_trade":
        return project_simplex_bounds(base, lo, hi)
    max_action = float(row["max_action_hint"])
    if max_action <= 0:
        return project_simplex_bounds(base, lo, hi)
    w = base.copy()
    idx = {s: i for i, s in enumerate(RISK_SLEEVES)}

    def move(amount: float, sources: list[str], targets: dict[str, float]) -> None:
        nonlocal w
        source_idx = [idx[s] for s in sources]
        target_idx = {idx[s]: v for s, v in targets.items()}
        available = np.maximum(w[source_idx] - lo[source_idx], 0.0)
        take = min(float(amount), float(available.sum()))
        if take <= 0:
            return
        w[source_idx] -= take * available / available.sum()
        total_target = sum(target_idx.values())
        for j, share in target_idx.items():
            w[j] += take * share / total_target

    gate = str(row["action_gate"])
    if gate == "allow_defensive_tilt":
        move(max_action, ["equity", "convertible_bonds", "credit"], {"rates": 0.70, "alternatives": 0.30})
    elif gate == "allow_risk_on_tilt":
        scores = {
            "equity": float(row["equity_trend_score"]),
            "convertible_bonds": float(row["cb_opportunity_score"]),
            "alternatives": max(float(row["commodity_inflation_score"]), float(row["alt_hedge_score"]) * 0.5),
        }
        positive = {k: max(v, 0.0) for k, v in scores.items()}
        if sum(positive.values()) <= 0:
            positive = {"equity": 0.4, "convertible_bonds": 0.4, "alternatives": 0.2}
        move(max_action, ["rates", "credit"], positive)
    elif gate == "allow_small_tilt":
        module_scores = {
            "rates": float(row["rate_bull_score"]),
            "convertible_bonds": float(row["cb_opportunity_score"]),
            "equity": float(row["equity_trend_score"]),
            "alternatives": max(float(row["alt_hedge_score"]), float(row["commodity_inflation_score"])),
        }
        target = max(module_scores, key=module_scores.get)
        if target == "rates":
            move(max_action, ["equity", "convertible_bonds", "credit"], {"rates": 1.0})
        elif target == "convertible_bonds":
            move(max_action, ["rates", "credit"], {"convertible_bonds": 1.0})
        elif target == "equity":
            move(max_action, ["rates", "credit"], {"equity": 1.0})
        elif target == "alternatives":
            move(max_action, ["rates", "credit", "equity"], {"alternatives": 1.0})
    return project_simplex_bounds(w, lo, hi)


def perf_stats(ret: pd.Series) -> dict[str, float]:
    ret = ret.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return {}
    nav = (1.0 + ret).cumprod()
    ann_ret = float(nav.iloc[-1] ** (252 / len(ret)) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else np.nan
    dd = nav / nav.cummax() - 1.0
    return {
        "days": int(len(ret)),
        "total_return": float(nav.iloc[-1] - 1.0),
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
    }


def main() -> None:
    timing_cfg = load_timing_card_config()
    overlay_cfg = load_overlay_config()
    cards = build_daily_timing_cards(timing_cfg)
    sleeve_ret = build_daily_sleeve_return_matrix(overlay_cfg).reset_index()
    rows = []
    daily_rows = []
    for profile in ["low_vol", "balanced", "high_vol"]:
        prof = cards[cards["profile"].eq(profile)].merge(sleeve_ret, on="date", how="inner").sort_values("date")
        lo, hi = bounds_for(profile, overlay_cfg)
        base_rets, timing_rets, turnovers = [], [], []
        decisions: list[np.ndarray] = []
        for _, row in prof.iterrows():
            base = np.array([float(row[f"current_weight_{s}"]) for s in RISK_SLEEVES], dtype=float)
            base = project_simplex_bounds(base, lo, hi)
            decisions.append(tilt_weights(row, base, lo, hi))

        prev_held = None
        for position, (_, row) in enumerate(prof.iterrows()):
            base = np.array([float(row[f"current_weight_{s}"]) for s in RISK_SLEEVES], dtype=float)
            base = project_simplex_bounds(base, lo, hi)
            # The card is formed after date t closes. Its decision is first held on t+1.
            timing = base if position == 0 else decisions[position - 1]
            r = np.array([float(row[s]) for s in RISK_SLEEVES], dtype=float)
            base_ret = float(np.dot(base, r))
            timing_ret_gross = float(np.dot(timing, r))
            turnover = 0.0 if prev_held is None else float(np.abs(timing - prev_held).sum())
            cost = turnover * float(overlay_cfg["reward"]["turnover_cost_bps"]) / 10000.0
            timing_ret = timing_ret_gross - cost
            prev_held = timing
            base_rets.append(base_ret)
            timing_rets.append(timing_ret)
            turnovers.append(turnover)
            daily_rows.append(
                {
                    "date": row["date"],
                    "signal_date": pd.NaT if position == 0 else prof.iloc[position - 1]["date"],
                    "profile": profile,
                    "base_return": base_ret,
                    "timing_return": timing_ret,
                    "turnover": turnover,
                    "action_gate": row["action_gate"],
                    "risk_regime": row["risk_regime"],
                }
            )
        df = pd.DataFrame({"date": prof["date"].to_numpy(), "base": base_rets, "timing": timing_rets, "turnover": turnovers})
        for split, mask in {
            "full": pd.Series(True, index=df.index),
            "oos_2024_onward": pd.to_datetime(df["date"]).ge(pd.Timestamp("2024-01-01")),
        }.items():
            base_stats = perf_stats(df.loc[mask, "base"])
            timing_stats = perf_stats(df.loc[mask, "timing"])
            rows.append(
                {
                    "profile": profile,
                    "split": split,
                    **{f"base_{k}": v for k, v in base_stats.items()},
                    **{f"timing_{k}": v for k, v in timing_stats.items()},
                    "delta_annual_return": timing_stats.get("annual_return", np.nan) - base_stats.get("annual_return", np.nan),
                    "delta_sharpe": timing_stats.get("sharpe", np.nan) - base_stats.get("sharpe", np.nan),
                    "timing_mean_turnover": float(df.loc[mask, "turnover"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)
    summary_path = ROOT / "output" / "reports" / "daily_timing_card_backtest_summary.csv"
    daily_path = ROOT / "output" / "reports" / "daily_timing_card_backtest_daily.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"summary": str(summary_path.relative_to(ROOT)), "daily": str(daily_path.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
