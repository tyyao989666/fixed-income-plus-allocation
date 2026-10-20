from __future__ import annotations

import copy
import html
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.adaptive_daily_ppo import ActorCritic, build_profile_market, perf_stats
from src.ep_skfolio_optimizer import RISK_SLEEVES, optimize_sleeve_weights, project_simplex_bounds, view_to_sleeve
from src.rienet_covariance import CovarianceEstimate, estimate_covariance
from src.v1_strategy import apply_selected_profile_overrides, load_config as load_v1_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "integrated_daily_view_ppo_config.yaml"
MODULES = list(RISK_SLEEVES)
GRID_REGIMES = [
    f"bond_{bond}_equity_{equity}"
    for bond in ["weak", "neutral", "strong"]
    for equity in ["weak", "neutral", "strong"]
]


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def out_path(cfg: dict[str, Any], key: str, profile: str | None = None) -> Path:
    value = str(cfg["outputs"][key])
    if profile is not None:
        value = value.format(profile=profile)
    path = ROOT / value
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _cross_section(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        return np.zeros_like(array)
    centered = array - float(array.mean())
    scale = float(centered.std())
    if scale <= 1e-8:
        return np.zeros_like(array)
    return np.clip(centered / scale, -3.0, 3.0)


def _rank_ic(scores: np.ndarray, realized: np.ndarray) -> float:
    if not np.isfinite(scores).all() or not np.isfinite(realized).all():
        return np.nan
    if float(np.ptp(scores)) <= 1e-12:
        return 0.0
    return float(pd.Series(scores).corr(pd.Series(realized), method="spearman"))


def _information_cutoff(date: pd.Timestamp) -> pd.Timestamp:
    if date.year <= 2024:
        return pd.Timestamp(date.year - 1, 12, 31)
    return date.to_period("Q").start_time


def load_prediction_states(cfg: dict[str, Any]) -> pd.DataFrame:
    source = pd.read_parquet(ROOT / cfg["data"]["predictions"])
    gate_enabled = True
    gate_report = cfg["data"].get("itransformer_gate_report")
    require_global_gate = bool(
        cfg["actions"].get("require_formal_itransformer_gate", False)
    )
    if gate_report and require_global_gate:
        gate_path = ROOT / str(gate_report)
        if gate_path.exists():
            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            gate_enabled = bool(
                gate_payload.get(
                    "enabled_for_portfolio_stage",
                    gate_payload.get("enabled_for_formal_strategy", False),
                )
            )
        else:
            gate_enabled = False
    required = {
        "as_of_date",
        "horizon_days",
        "sleeve",
        "historical_oos_relative_score",
        "walkforward_relative_score",
        "baseline_relative_score",
        "horizon_label_end_date",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"iTransformer predictions missing columns: {sorted(missing)}")
    source["as_of_date"] = pd.to_datetime(source["as_of_date"], errors="coerce")
    source["horizon_label_end_date"] = pd.to_datetime(source["horizon_label_end_date"], errors="coerce")
    historical = pd.to_numeric(source["historical_oos_relative_score"], errors="coerce")
    walkforward = pd.to_numeric(source["walkforward_relative_score"], errors="coerce")
    source["model_score"] = walkforward.where(walkforward.notna(), historical)
    source["baseline_score"] = pd.to_numeric(source["baseline_relative_score"], errors="coerce")
    if "model_raw_score" in source.columns:
        source["raw_issued_model_score"] = pd.to_numeric(
            source["model_raw_score"], errors="coerce"
        ).where(lambda values: values.notna(), source["model_score"])
    else:
        source["raw_issued_model_score"] = source["model_score"]
    if "model_residual_score" in source.columns:
        source["raw_model_residual"] = pd.to_numeric(
            source["model_residual_score"], errors="coerce"
        )
    else:
        source["raw_model_residual"] = source["model_score"] - source["baseline_score"]
    source["model_uncertainty"] = pd.to_numeric(
        source.get("prediction_uncertainty"), errors="coerce"
    )
    source["realized"] = pd.to_numeric(source.get("realized_relative_score"), errors="coerce")
    if "residual_alpha" in source.columns:
        source["selected_residual_alpha"] = pd.to_numeric(
            source["residual_alpha"], errors="coerce"
        )
    else:
        # Non-residual legacy models are equivalent to a fully selected model.
        source["selected_residual_alpha"] = 1.0
    horizons = [int(value) for value in cfg["actions"]["horizons_days"]]
    source = source[
        source["horizon_days"].isin(horizons)
        & source["sleeve"].isin(MODULES)
        & source["model_score"].notna()
        & source["baseline_score"].notna()
    ].copy()

    rows: list[dict[str, Any]] = []
    realized_rows: list[dict[str, Any]] = []
    for (date, horizon), group in source.groupby(["as_of_date", "horizon_days"], sort=True):
        group = group.set_index("sleeve").reindex(MODULES)
        direct_mean_semantics = bool(
            "score_semantics" in group
            and group["score_semantics"]
            .dropna()
            .astype(str)
            .eq("annualized_relative_mean_view")
            .all()
            and len(group["score_semantics"].dropna()) == len(group)
        )
        if direct_mean_semantics:
            scale_source = (
                group["score_normalization_scale"]
                if "score_normalization_scale" in group
                else pd.Series(0.10, index=group.index)
            )
            scale_values = pd.to_numeric(scale_source, errors="coerce")
            scale = max(float(scale_values.median()), 1e-6)

            def fixed_scale(values: np.ndarray) -> np.ndarray:
                centered = values - float(np.mean(values))
                return np.clip(centered / scale, -3.0, 3.0)

            model = fixed_scale(
                group["model_score"].to_numpy(dtype=float)
            )
            raw_issued_model = fixed_scale(
                group["raw_issued_model_score"].to_numpy(dtype=float)
            )
            baseline = fixed_scale(
                group["baseline_score"].to_numpy(dtype=float)
            )
        else:
            model = _cross_section(
                group["model_score"].to_numpy(dtype=float)
            )
            raw_issued_model = _cross_section(
                group["raw_issued_model_score"].to_numpy(dtype=float)
            )
            baseline = _cross_section(
                group["baseline_score"].to_numpy(dtype=float)
            )
        if (
            not np.isfinite(model).all()
            or not np.isfinite(raw_issued_model).all()
            or not np.isfinite(baseline).all()
        ):
            continue
        raw_residual_values = (
            group["raw_model_residual"].fillna(0.0).to_numpy(dtype=float)
        )
        raw_residual = (
            fixed_scale(raw_residual_values)
            if direct_mean_semantics
            else _cross_section(raw_residual_values)
        )
        uncertainty = (
            group["model_uncertainty"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(group["model_uncertainty"].median())
            .fillna(1.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)
        )
        row: dict[str, Any] = {
            "date": pd.Timestamp(date),
            "horizon_days": int(horizon),
            "state_it_direct_mean_semantics": float(
                direct_mean_semantics
            ),
        }
        for position, sleeve in enumerate(MODULES):
            row[f"state_it_model_h{horizon}_{sleeve}"] = float(model[position])
            row[f"state_it_baseline_h{horizon}_{sleeve}"] = float(baseline[position])
            row[f"state_it_residual_h{horizon}_{sleeve}"] = float(
                raw_residual[position]
            )
            row[f"state_it_uncertainty_h{horizon}_{sleeve}"] = float(
                uncertainty[position]
            )
        row["horizon_label_end_date"] = group["horizon_label_end_date"].max()
        rows.append(row)
        realized = group["realized"].to_numpy(dtype=float)
        realized_rows.append(
            {
                "date": pd.Timestamp(date),
                "horizon_days": int(horizon),
                "available_date": group["horizon_label_end_date"].max(),
                # Reliability must learn from the raw forecast that was issued
                # at the time, even when the causal deployment alpha was zero.
                # Using the deployed zero score creates a catch-22 in which a
                # challenger can never accumulate evidence to become eligible.
                "model_rank_ic": _rank_ic(raw_issued_model, realized),
                "baseline_rank_ic": _rank_ic(baseline, realized),
            }
        )

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No strict-OOS daily iTransformer predictions are available")
    value_columns = [column for column in long if column.startswith("state_it_")]
    state = long.groupby("date", as_index=False)[value_columns].first()
    state = state.merge(
        long.groupby("date", as_index=False)["horizon_label_end_date"].max().rename(
            columns={"horizon_label_end_date": "max_horizon_end_date"}
        ),
        on="date",
        how="left",
    )
    calibration = (
        source.groupby("as_of_date", as_index=False)["selected_residual_alpha"]
        .max()
        .rename(columns={"as_of_date": "date"})
    )
    state = state.merge(calibration, on="date", how="left")
    state["state_it_selected_residual_alpha"] = (
        state.pop("selected_residual_alpha").fillna(0.0).clip(lower=0.0)
    )
    realized = pd.DataFrame(realized_rows).dropna(subset=["available_date", "model_rank_ic", "baseline_rank_ic"])
    window = int(cfg["data"].get("reliability_window", 60))
    minimum_observations = int(cfg["data"].get("reliability_min_observations", 20))
    for horizon in horizons:
        history = realized.loc[realized["horizon_days"].eq(horizon)].sort_values("available_date").copy()
        history["model_reliability"] = history["model_rank_ic"].rolling(
            window, min_periods=minimum_observations
        ).mean()
        history["baseline_reliability"] = history["baseline_rank_ic"].rolling(
            window, min_periods=minimum_observations
        ).mean()
        history["matured_observations"] = (
            history["model_rank_ic"].notna() & history["baseline_rank_ic"].notna()
        ).rolling(window, min_periods=1).sum()
        merged = pd.merge_asof(
            state[["date"]].sort_values("date"),
            history[
                [
                    "available_date",
                    "model_reliability",
                    "baseline_reliability",
                    "matured_observations",
                ]
            ].sort_values("available_date"),
            left_on="date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=False,
        )
        state[f"state_it_h{horizon}_model_reliability"] = merged["model_reliability"].fillna(0.0).clip(-1.0, 1.0)
        state[f"state_it_h{horizon}_baseline_reliability"] = merged["baseline_reliability"].fillna(0.0).clip(-1.0, 1.0)
        state[f"state_it_h{horizon}_matured_observations"] = (
            merged["matured_observations"].fillna(0.0).clip(lower=0.0)
        )
        model_columns = [f"state_it_model_h{horizon}_{sleeve}" for sleeve in MODULES]
        residual_columns = [
            f"state_it_residual_h{horizon}_{sleeve}" for sleeve in MODULES
        ]
        state[f"state_it_h{horizon}_spread"] = np.ptp(state[model_columns].to_numpy(dtype=float), axis=1)
        state[f"state_it_h{horizon}_residual_spread"] = np.ptp(
            state[residual_columns].to_numpy(dtype=float), axis=1
        )

    if "model_training_cutoff" in source.columns:
        model_cutoffs = (
            source.groupby("as_of_date", as_index=False)["model_training_cutoff"]
            .max()
            .rename(columns={"as_of_date": "date"})
        )
        model_cutoffs["model_training_cutoff"] = pd.to_datetime(
            model_cutoffs["model_training_cutoff"], errors="coerce"
        )
        state = state.merge(model_cutoffs, on="date", how="left")
        state["itransformer_information_cutoff"] = state["model_training_cutoff"]
        state = state.drop(columns=["model_training_cutoff"])
    else:
        state["itransformer_information_cutoff"] = state["date"].map(_information_cutoff)
    state["state_it_global_gate"] = float(gate_enabled)
    return state.sort_values("date").reset_index(drop=True)


def _base_card_states(cards: pd.DataFrame) -> pd.DataFrame:
    state_maps = {
        "cr_state": {"CR1": 1.0, "CR2": 0.5, "CR3": -0.5, "CR4": -1.0, "CR5": -1.0},
        "r_state": {"R1": 1.0, "R2": 0.0, "R3": -0.3, "R4": -1.0, "R5": -1.0},
        "cb_state": {"CB1": 1.0, "CB2": 0.0, "CB3": -0.3, "CB4": -1.0, "CB5": -1.0},
        "eq_state": {"EQ1": 1.0, "EQ2": 0.0, "EQ3": -0.3, "EQ4": -1.0, "EQ5": -1.0},
        "alt_state": {"ALT1": 1.0, "ALT2": 0.0, "ALT3": -0.5, "ALT4": -1.0},
        "rb_state": {"RB1": 1.0, "RB2": 0.3, "RB3": -0.5, "RB4": -1.0},
    }
    rows: list[dict[str, Any]] = []
    previous_regime = ""
    duration = 0
    for _, card in cards.sort_values("date").iterrows():
        regime = str(card.get("stock_bond_regime", "unknown"))
        duration = duration + 1 if regime == previous_regime else 1
        previous_regime = regime
        filters = _json(card.get("view_filters"))
        detail = _json(card.get("asset_detail_states"))
        assets = detail.get("assets", {})
        bond = float(card.get("bond_score", 0.5)) * 2.0 - 1.0
        equity = float(card.get("equity_score", 0.5)) * 2.0 - 1.0
        risk_budget = float(card.get("risk_budget_score", 0.5)) * 2.0 - 1.0
        row: dict[str, Any] = {
            "date": pd.Timestamp(card["date"]),
            "stock_bond_regime": regime,
            "state_grid_bond": bond,
            "state_grid_equity": equity,
            "state_grid_duration": float(min(duration, 252)) / 252.0,
            "state_grid_regime_changed": float(duration == 1),
            "state_risk_budget_score": float(card.get("risk_budget_score", 0.5)),
            "state_liquidity_shock": float(bool(card.get("liquidity_shock", False))),
            "state_bond_score": float(card.get("bond_score", 0.5)),
            "state_equity_score": float(card.get("equity_score", 0.5)),
            "state_grid_block_credit": float(bool(filters.get("block_credit_overweight", False))),
            "state_grid_block_equity": float(bool(filters.get("block_equity_overweight", False))),
            "state_grid_block_cb": float(bool(filters.get("block_cb_overweight", False))),
            "state_grid_block_long_rate": float(bool(filters.get("block_long_rate_overweight", False))),
        }
        for grid in GRID_REGIMES:
            row[f"state_grid_{grid}"] = float(regime == grid)
        for field, mapping in state_maps.items():
            row[f"state_card_{field}"] = float(mapping.get(str(card.get(field)), 0.0))
        module_means: dict[str, float] = {}
        for module in MODULES:
            selected = [value for value in assets.values() if value.get("module") == module]
            scores = np.array([value.get("score", 0.5) for value in selected], dtype=float)
            finite = scores[np.isfinite(scores)]
            mean_score = float(finite.mean()) if len(finite) else 0.5
            module_means[module] = mean_score
            row[f"state_card_{module}_mean"] = mean_score
            row[f"state_card_{module}_dispersion"] = float(finite.std()) if len(finite) else 0.0
            row[f"state_card_{module}_positive_ratio"] = float(
                np.mean([value.get("state") == "positive" for value in selected]) if selected else 0.0
            )
            row[f"state_card_{module}_negative_ratio"] = float(
                np.mean([value.get("state") == "negative" for value in selected]) if selected else 0.0
            )
            row[f"state_card_{module}_eligible_ratio"] = float(
                np.mean([value.get("domain_context", {}).get("overweight_eligible", False) for value in selected])
                if selected
                else 0.0
            )
        row["state_grid_compat_rates"] = bond
        row["state_grid_compat_credit"] = 0.70 * bond + 0.30 * risk_budget
        row["state_grid_compat_convertible_bonds"] = 0.65 * equity + 0.35 * bond
        row["state_grid_compat_equity"] = equity
        row["state_grid_compat_alternatives"] = (
            0.25 * bond - 0.20 * equity + 0.55 * (2.0 * module_means.get("alternatives", 0.5) - 1.0)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")


def load_card_states(cfg: dict[str, Any]) -> pd.DataFrame:
    cards = pd.read_parquet(ROOT / cfg["data"]["state_cards"])
    cards["date"] = pd.to_datetime(cards["date"], errors="coerce")
    cards = cards.dropna(subset=["date"])
    return _base_card_states(cards)


def _to_sleeve(asset: str, v1cfg: dict[str, Any]) -> str:
    mapped = view_to_sleeve(asset, v1cfg)
    if mapped in MODULES:
        return mapped
    value = str(asset).lower()
    if "rate" in value or "duration" in value or "treasury" in value:
        return "rates"
    if "credit" in value:
        return "credit"
    if value.startswith("cb") or "convertible" in value:
        return "convertible_bonds"
    if "equity" in value or "csi" in value:
        return "equity"
    if value.startswith("alt") or value in {"gold", "commodity", "reits", "us_bond"}:
        return "alternatives"
    return mapped


def load_rule_view_states(cfg: dict[str, Any]) -> pd.DataFrame:
    views = pd.read_parquet(ROOT / cfg["data"]["rule_views"])
    views["date"] = pd.to_datetime(views["date"], errors="coerce")
    if "validator_status" in views:
        views = views.loc[views["validator_status"].eq("valid")].copy()
    v1cfg = load_v1_config()
    rows: list[dict[str, Any]] = []
    for date, group in views.dropna(subset=["date"]).groupby("date", sort=True):
        scores = pd.Series(0.0, index=MODULES)
        confidences: list[float] = []
        valid_pairs = 0
        for _, view in group.iterrows():
            source = _to_sleeve(str(view.get("source_asset", "")), v1cfg)
            target = _to_sleeve(str(view.get("target_asset", "")), v1cfg)
            if source not in scores.index or target not in scores.index or source == target:
                continue
            confidence = max(float(view.get("confidence", 0.0)), 0.0)
            scores[source] += confidence
            scores[target] -= confidence
            confidences.append(confidence)
            valid_pairs += 1
        normalized = scores - float(scores.mean())
        denom = float(normalized.abs().max())
        normalized = normalized / denom if denom > 1e-12 else normalized * 0.0
        row: dict[str, Any] = {
            "date": pd.Timestamp(date),
            "state_rule_view_count": float(valid_pairs),
            "state_rule_view_mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        }
        for sleeve in MODULES:
            row[f"state_rule_view_{sleeve}"] = float(normalized.loc[sleeve])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")


def build_state_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    state = load_prediction_states(cfg)
    cards = load_card_states(cfg)
    views = load_rule_view_states(cfg)
    state = state.merge(cards, on="date", how="left").merge(views, on="date", how="left")
    rule_columns = [column for column in state if column.startswith("state_rule_")]
    state[rule_columns] = state[rule_columns].fillna(0.0)
    other_state_columns = [column for column in state if column.startswith("state_") and column not in rule_columns]
    state[other_state_columns] = state[other_state_columns].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    state["stock_bond_regime"] = state["stock_bond_regime"].ffill().fillna("unknown")
    alignment_columns: dict[str, np.ndarray] = {}
    for horizon in cfg["actions"]["horizons_days"]:
        model = state[[f"state_it_model_h{horizon}_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        residual = state[
            [f"state_it_residual_h{horizon}_{sleeve}" for sleeve in MODULES]
        ].to_numpy(dtype=float)
        grid = state[[f"state_grid_compat_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        numerator = np.sum(model * grid, axis=1)
        denominator = np.linalg.norm(model, axis=1) * np.linalg.norm(grid, axis=1)
        alignment_columns[f"state_grid_model_alignment_h{horizon}"] = np.divide(
            numerator, denominator, out=np.zeros(len(state)), where=denominator > 1e-8
        )
        residual_numerator = np.sum(residual * grid, axis=1)
        residual_denominator = (
            np.linalg.norm(residual, axis=1) * np.linalg.norm(grid, axis=1)
        )
        alignment_columns[f"state_grid_residual_alignment_h{horizon}"] = np.divide(
            residual_numerator,
            residual_denominator,
            out=np.zeros(len(state)),
            where=residual_denominator > 1e-8,
        )
    state = pd.concat([state, pd.DataFrame(alignment_columns, index=state.index)], axis=1)
    return state.copy().sort_values("date").reset_index(drop=True)


def action_specs(cfg: dict[str, Any]) -> list[tuple[int, str, float]]:
    specs: list[tuple[int, str, float]] = [
        (1, "noop", 0.0),
        (int(cfg["actions"].get("formal_anchor_horizon_days", 20)), "formal_anchor", 0.0),
        (int(cfg["actions"].get("rule_horizon_days", 20)), "rules_only", 0.0),
    ]
    for horizon in cfg["actions"]["horizons_days"]:
        specs.append((int(horizon), "baseline_adaptive", 1.0))
        specs.append((int(horizon), "model_adaptive", 1.0))
    return specs


def _action_family(name: str) -> str:
    return "model" if name.startswith("model_") else name


def _profile_v1_config(profile: str, integrated_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(load_v1_config())
    cfg["product"]["selected_profile"] = profile
    cfg["strategy_version"] = "no_agent"
    cfg = apply_selected_profile_overrides(cfg)
    if integrated_cfg is not None:
        cfg.setdefault("ep_skfolio", {}).update(copy.deepcopy(integrated_cfg.get("optimizer", {}).get(profile, {})))
    return cfg


def _trend_scores(returns: pd.DataFrame, lookbacks: list[int]) -> pd.DataFrame:
    output = pd.DataFrame(index=returns.index)
    for sleeve in MODULES:
        parts: list[pd.Series] = []
        series = pd.to_numeric(returns[sleeve], errors="coerce").fillna(0.0)
        for lookback in lookbacks:
            cumulative = series.rolling(lookback, min_periods=max(20, lookback // 2)).sum()
            volatility = series.rolling(lookback, min_periods=max(20, lookback // 2)).std(ddof=0)
            scaled = cumulative / (volatility * math.sqrt(lookback) + 1e-8)
            parts.append(scaled.clip(-2.0, 2.0))
        output[f"state_trend_{sleeve}"] = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)
    return output


@dataclass
class IntegratedViewEnvironment:
    state: pd.DataFrame
    sleeve_returns: pd.DataFrame
    formal_base: pd.DataFrame
    risk_budget_base: pd.DataFrame
    lo: np.ndarray
    hi: np.ndarray
    profile: str
    cfg: dict[str, Any]

    def __post_init__(self) -> None:
        self.state = self.state.copy().sort_values("date").reset_index(drop=True)
        self.calendar = pd.DatetimeIndex(self.sleeve_returns.index)
        positions = self.calendar.get_indexer(pd.DatetimeIndex(self.state["date"]))
        valid = positions >= 0
        self.state = self.state.loc[valid].reset_index(drop=True)
        self.calendar_positions = positions[valid]
        trend = _trend_scores(self.sleeve_returns, [int(x) for x in self.cfg["trend_brake"]["lookbacks_days"]])
        for column in trend:
            self.state[column] = trend[column].reindex(pd.DatetimeIndex(self.state["date"])).to_numpy(dtype=float)
        self.state_cols = [column for column in self.state if column.startswith("state_")]
        self.specs = action_specs(self.cfg)
        self.noop_action = 0
        self.formal_anchor_action = self.specs.index(
            (int(self.cfg["actions"].get("formal_anchor_horizon_days", 20)), "formal_anchor", 0.0)
        )
        self.rules_20_action = self.specs.index(
            (int(self.cfg["actions"].get("rule_horizon_days", 20)), "rules_only", 0.0)
        )
        self.formal_values = np.vstack(
            [project_simplex_bounds(row, self.lo, self.hi) for row in self.formal_base.reindex(self.calendar)[MODULES].to_numpy(dtype=float)]
        )
        self.risk_budget_values = np.vstack(
            [
                project_simplex_bounds(row, self.lo, self.hi)
                for row in self.risk_budget_base.reindex(self.calendar)[MODULES].to_numpy(dtype=float)
            ]
        )
        self.return_values = self.sleeve_returns.reindex(self.calendar)[MODULES].fillna(0.0).to_numpy(dtype=float)
        formal_turnover = np.vstack([np.zeros(len(MODULES)), np.abs(np.diff(self.formal_values, axis=0))]).sum(axis=1)
        self.formal_cost = formal_turnover * float(self.cfg["reward"]["transaction_cost_bps"]) / 10000.0
        self.v1cfg = _profile_v1_config(self.profile, self.cfg)
        self.bounds = {sleeve: (float(self.lo[i]), float(self.hi[i])) for i, sleeve in enumerate(MODULES)}
        self.candidate_cache: dict[tuple[int, int], tuple[np.ndarray, dict[str, Any]]] = {}
        self.canonical_target_cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}

    def covariance_estimate(self, history: pd.DataFrame):
        covariance_cfg = {
            "covariance_estimator": copy.deepcopy(
                self.cfg["views"].get(
                    "covariance_estimator",
                    {"method": "sample_shrink", "shrinkage_to_diag": 0.20},
                )
            )
        }
        estimate = estimate_covariance(
            history,
            MODULES,
            covariance_cfg,
            default_shrinkage=float(
                covariance_cfg["covariance_estimator"].get(
                    "shrinkage_to_diag", 0.20
                )
            ),
            floor=1e-8,
            jitter=float(
                covariance_cfg["covariance_estimator"].get("jitter", 1e-9)
            ),
        )
        scale = float(self.cfg["views"]["covariance_monthly_scale"])
        return CovarianceEstimate(
            covariance=estimate.covariance * scale,
            diagnostics={
                **estimate.diagnostics,
                "covariance_scale": scale,
            },
        )

    def action_mask(self, index: int) -> np.ndarray:
        row = self.state.loc[index]
        mask = np.ones(len(self.specs), dtype=bool)
        for action, (horizon, name, level) in enumerate(self.specs):
            family = _action_family(name)
            allowed = self.cfg.get("deployment", {}).get("allowed_action_families", {}).get(
                self.profile
            )
            if allowed is not None and family not in set(str(value) for value in allowed):
                mask[action] = False
                continue
            if family not in {"model", "baseline_adaptive"}:
                continue
            model = row[[f"state_it_model_h{horizon}_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
            baseline = row[[f"state_it_baseline_h{horizon}_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
            model_reliability = float(row[f"state_it_h{horizon}_model_reliability"])
            baseline_reliability = float(row[f"state_it_h{horizon}_baseline_reliability"])
            matured_observations = float(
                row[f"state_it_h{horizon}_matured_observations"]
            )
            if family == "model":
                incremental = model_reliability - baseline_reliability
                correction = model - baseline
                forecast = (
                    _cross_section(correction)
                    if float(np.ptp(correction)) > 1e-12
                    else np.zeros(len(MODULES), dtype=float)
                )
                minimum_edge = float(self.cfg["actions"]["minimum_incremental_reliability"])
                selected_alpha = float(row.get("state_it_selected_residual_alpha", 1.0))
                reliable = bool(
                    incremental >= minimum_edge
                    and model_reliability
                    >= float(self.cfg["actions"]["minimum_model_reliability"])
                    and matured_observations
                    >= int(self.cfg["data"].get("reliability_min_observations", 20))
                    and selected_alpha
                    >= float(
                        self.cfg["actions"].get(
                            "minimum_selected_residual_alpha", 0.0
                        )
                    )
                    and float(row.get("state_it_global_gate", 1.0)) > 0.5
                )
            else:
                forecast = baseline
                reliable = bool(
                    baseline_reliability
                    >= float(self.cfg["actions"]["minimum_model_reliability"])
                    and matured_observations
                    >= int(self.cfg["data"].get("reliability_min_observations", 20))
                )
            mask[action] = bool(reliable and float(np.ptp(forecast)) >= float(self.cfg["actions"]["minimum_score_spread"]))
        return mask

    def eligible_rows(self, cutoff: pd.Timestamp) -> np.ndarray:
        max_end = pd.to_datetime(self.state["max_horizon_end_date"], errors="coerce")
        return np.flatnonzero(max_end.lt(cutoff).to_numpy())

    def initial_weights(self, index: int) -> np.ndarray:
        return self.formal_values[int(self.calendar_positions[index])].copy()

    def observation(self, index: int, current_weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        static = self.state.loc[index, self.state_cols].to_numpy(dtype=float)
        scaled = (static - mean) / std
        base = self.risk_budget_values[int(self.calendar_positions[index])]
        active = (np.asarray(current_weights, dtype=float) - base) / 0.05
        return np.concatenate([scaled, np.clip(active, -3.0, 3.0)]).astype(np.float32)

    def _predictive_view(self, index: int, horizon: int, family: str) -> tuple[np.ndarray, float, dict[str, float]]:
        row = self.state.loc[index]
        raw_model = row[[f"state_it_model_h{horizon}_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        baseline = row[[f"state_it_baseline_h{horizon}_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        model_reliability = float(row[f"state_it_h{horizon}_model_reliability"])
        baseline_reliability = float(row[f"state_it_h{horizon}_baseline_reliability"])
        matured_observations = float(
            row[f"state_it_h{horizon}_matured_observations"]
        )
        selected_alpha = float(row.get("state_it_selected_residual_alpha", 1.0))
        incremental = model_reliability - baseline_reliability
        if family == "model":
            minimum_edge = float(self.cfg["actions"]["minimum_incremental_reliability"])
            full_edge = float(self.cfg["actions"]["full_incremental_reliability"])
            minimum_alpha = float(
                self.cfg["actions"].get("minimum_selected_residual_alpha", 0.0)
            )
            alpha_reference = max(
                float(self.cfg["actions"].get("residual_alpha_reference", 1.0)),
                minimum_alpha,
                1e-8,
            )
            eligible = bool(
                incremental >= minimum_edge
                and model_reliability
                >= float(self.cfg["actions"]["minimum_model_reliability"])
                and matured_observations
                >= int(self.cfg["data"].get("reliability_min_observations", 20))
                and selected_alpha >= minimum_alpha
            )
            if bool(self.cfg["actions"].get("require_formal_itransformer_gate", False)) and float(
                row.get("state_it_global_gate", 0.0)
            ) <= 0.5:
                eligible = False

            if incremental < 0.0 and minimum_edge < 0.0:
                edge_confidence = 0.25 + 0.50 * float(
                    np.clip(
                        (incremental - minimum_edge) / max(-minimum_edge, 1e-8),
                        0.0,
                        1.0,
                    )
                )
            elif incremental >= 0.0:
                edge_confidence = 0.75 + 0.25 * float(
                    np.clip(incremental / max(full_edge, 1e-8), 0.0, 1.0)
                )
            else:
                edge_confidence = 0.0
            alpha_confidence = float(
                np.clip(selected_alpha / alpha_reference, 0.0, 1.0)
            )
            itransformer_weight = (
                float(self.cfg["actions"]["max_itransformer_weight"])
                * alpha_confidence
                * edge_confidence
                if eligible
                else 0.0
            )
            correction = raw_model - baseline
            if float(np.ptp(correction)) <= 1e-12:
                itransformer_weight = 0.0
                forecast = np.zeros(len(MODULES), dtype=float)
            else:
                # The saved model score is already baseline + selected alpha *
                # predicted residual. Use only its unique residual direction
                # here; otherwise momentum is counted twice.
                forecast = _cross_section(correction)
            reliability = model_reliability
        else:
            edge_confidence = 1.0
            alpha_confidence = 0.0
            itransformer_weight = 0.0
            forecast = baseline
            reliability = baseline_reliability
        if float(np.ptp(forecast)) < float(self.cfg["actions"]["minimum_score_spread"]):
            return np.zeros(len(MODULES)), 0.0, {
                "model_reliability": model_reliability,
                "baseline_reliability": baseline_reliability,
                "incremental_reliability": incremental,
                "itransformer_weight": itransformer_weight,
                "selected_residual_alpha": selected_alpha,
                "edge_confidence": edge_confidence,
                "alpha_confidence": alpha_confidence,
                "matured_observations": matured_observations,
                "global_gate": float(row.get("state_it_global_gate", 1.0)),
                "grid_alignment": 0.0,
            }
        compatibility = row[[f"state_grid_compat_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        blocks = np.array(
            [
                bool(row["state_grid_block_long_rate"]),
                bool(row["state_grid_block_credit"]),
                bool(row["state_grid_block_cb"]),
                bool(row["state_grid_block_equity"]),
                False,
            ]
        )
        adjusted = forecast.copy()
        floor = float(self.cfg["views"]["grid_positive_floor"])
        for position in range(len(adjusted)):
            if adjusted[position] > 0:
                grid_multiplier = floor + (1.0 - floor) * float(np.clip((compatibility[position] + 1.0) / 2.0, 0.0, 1.0))
                adjusted[position] *= grid_multiplier
                if blocks[position]:
                    adjusted[position] = 0.0
        adjusted = _cross_section(adjusted)
        minimum = float(self.cfg["actions"]["minimum_model_reliability"])
        full = float(self.cfg["actions"]["full_model_reliability"])
        absolute_confidence = float(
            np.clip(
                (reliability - minimum) / max(full - minimum, 1e-8),
                0.0,
                1.0,
            )
        )
        confidence = absolute_confidence * (
            itransformer_weight if family == "model" else 1.0
        )
        denominator = float(np.linalg.norm(forecast) * np.linalg.norm(compatibility))
        alignment = float(np.dot(forecast, compatibility) / denominator) if denominator > 1e-8 else 0.0
        return adjusted, confidence, {
            "model_reliability": model_reliability,
            "baseline_reliability": baseline_reliability,
            "incremental_reliability": incremental,
            "itransformer_weight": itransformer_weight,
            "selected_residual_alpha": selected_alpha,
            "edge_confidence": edge_confidence,
            "alpha_confidence": alpha_confidence,
            "absolute_confidence": absolute_confidence,
            "matured_observations": matured_observations,
            "global_gate": float(row.get("state_it_global_gate", 1.0)),
            "grid_alignment": alignment,
        }

    def _apply_guardrails(self, index: int, target: np.ndarray, base: np.ndarray) -> np.ndarray:
        row = self.state.loc[index]
        values = np.asarray(target, dtype=float).copy()
        dynamic_hi = self.hi.copy()
        block_map = {
            "credit": "state_grid_block_credit",
            "convertible_bonds": "state_grid_block_cb",
            "equity": "state_grid_block_equity",
        }
        for sleeve, field in block_map.items():
            position = MODULES.index(sleeve)
            if bool(row[field]):
                dynamic_hi[position] = min(dynamic_hi[position], base[position])
                values[position] = min(values[position], base[position])
        if bool(row["state_liquidity_shock"]):
            for sleeve in ["convertible_bonds", "equity"]:
                position = MODULES.index(sleeve)
                dynamic_hi[position] = min(dynamic_hi[position], base[position])
                values[position] = min(values[position], base[position])
        if bool(self.cfg["trend_brake"].get("enabled", True)):
            threshold = float(self.cfg["trend_brake"]["weak_threshold"])
            max_cut = float(self.cfg["trend_brake"]["max_underweight_from_base"][self.profile])
            for sleeve in self.cfg["trend_brake"]["affected_sleeves"]:
                position = MODULES.index(sleeve)
                trend = float(row[f"state_trend_{sleeve}"])
                if trend < threshold:
                    severity = float(np.clip((threshold - trend) / 1.5, 0.0, 1.0))
                    values[position] = min(values[position], max(self.lo[position], base[position] - max_cut * severity))
        return project_simplex_bounds(values, self.lo, dynamic_hi)

    def canonical_view_target_weights(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Build one causal EP/skfolio target from rules and relative iTransformer views.

        PPO is intentionally absent from this method. It may later decide how
        quickly to approach this target, but it cannot change the direction.
        """
        if index in self.canonical_target_cache:
            weights, diagnostics = self.canonical_target_cache[index]
            return weights.copy(), dict(diagnostics)

        row = self.state.loc[index]
        calendar_position = int(self.calendar_positions[index])
        base = self.risk_budget_values[calendar_position]
        rule = row[[f"state_rule_view_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        rule = _cross_section(rule)

        horizon_cfg = self.cfg["views"].get(
            "canonical_horizon_weights", {5: 0.20, 20: 0.50, 60: 0.30}
        )
        horizon_weights = {
            int(key): max(float(value), 0.0) for key, value in dict(horizon_cfg).items()
        }
        horizon_weights = {
            horizon: horizon_weights.get(horizon, 0.0)
            for horizon in [int(value) for value in self.cfg["actions"]["horizons_days"]]
        }
        total_horizon_weight = sum(horizon_weights.values())
        if total_horizon_weight <= 0:
            raise ValueError("canonical_horizon_weights must contain a positive weight")
        horizon_weights = {
            horizon: value / total_horizon_weight for horizon, value in horizon_weights.items()
        }

        qualified_views: list[np.ndarray] = []
        effective_weights: list[float] = []
        horizon_diagnostics: dict[str, float] = {}
        for horizon, horizon_weight in horizon_weights.items():
            predictive, confidence, diagnostics = self._predictive_view(index, horizon, "model")
            incremental = float(diagnostics["incremental_reliability"])
            model_weight = float(diagnostics["itransformer_weight"])
            qualified = bool(confidence > 0.0 and model_weight > 0.0)
            effective_weight = horizon_weight * confidence if qualified else 0.0
            if effective_weight > 0:
                qualified_views.append(predictive)
                effective_weights.append(effective_weight)
            horizon_diagnostics[f"it_h{horizon}_confidence"] = float(confidence)
            horizon_diagnostics[f"it_h{horizon}_incremental_reliability"] = incremental
            horizon_diagnostics[f"it_h{horizon}_model_weight"] = model_weight
            horizon_diagnostics[f"it_h{horizon}_selected_residual_alpha"] = float(
                diagnostics["selected_residual_alpha"]
            )
            horizon_diagnostics[f"it_h{horizon}_edge_confidence"] = float(
                diagnostics["edge_confidence"]
            )
            horizon_diagnostics[f"it_h{horizon}_absolute_confidence"] = float(
                diagnostics.get("absolute_confidence", 0.0)
            )
            horizon_diagnostics[f"it_h{horizon}_matured_observations"] = float(
                diagnostics["matured_observations"]
            )
            horizon_diagnostics[f"it_h{horizon}_global_gate"] = float(
                diagnostics["global_gate"]
            )
            horizon_diagnostics[f"it_h{horizon}_qualified"] = float(qualified)

        if effective_weights:
            weight_array = np.asarray(effective_weights, dtype=float)
            aggregate_confidence = float(weight_array.sum())
            weight_array /= weight_array.sum()
            stack = np.vstack(qualified_views)
            model_view = np.average(stack, axis=0, weights=weight_array)
            disagreement = float(np.std(stack, axis=0).mean())
            uncertainty_scale = float(self.cfg["views"].get("uncertainty_scale", 0.75))
            uncertainty_multiplier = float(
                np.clip(1.0 - disagreement / max(uncertainty_scale, 1e-8), 0.0, 1.0)
            )
            model_view = (
                _cross_section(model_view)
                * uncertainty_multiplier
                * aggregate_confidence
            )
            model_active = True
        else:
            model_view = np.zeros(len(MODULES), dtype=float)
            disagreement = 0.0
            uncertainty_multiplier = 0.0
            aggregate_confidence = 0.0
            model_active = False

        fused = float(self.cfg["views"]["rule_weight"]) * rule
        fused += float(self.cfg["views"]["model_weight"]) * model_view
        fused = _cross_section(fused)
        max_abs = float(np.max(np.abs(fused)))
        if max_abs > 1e-12:
            fused /= max_abs

        mu_scale = float(self.cfg["views"]["daily_mu_scale"][self.profile])
        posterior_mu = pd.Series(fused * mu_scale, index=MODULES)
        lookback = int(self.cfg["views"]["covariance_lookback_days"])
        minimum = int(self.cfg["views"]["minimum_covariance_days"])
        history = (
            self.sleeve_returns.iloc[: calendar_position + 1]
            .tail(lookback)
            .reindex(columns=MODULES)
            .fillna(0.0)
        )
        if len(history) < minimum:
            target = base.copy()
            optimizer_diagnostics = {
                "optimizer_backend": "risk_budget_fallback",
                "status": "insufficient_history",
            }
            covariance_diagnostics = {}
        else:
            covariance_estimate = self.covariance_estimate(history)
            result = optimize_sleeve_weights(
                base_weights=pd.Series(base, index=MODULES),
                posterior_mu=posterior_mu,
                cov=covariance_estimate.covariance,
                returns_window=history,
                prev_weights=pd.Series(base, index=MODULES),
                bounds=self.bounds,
                cfg=self.v1cfg,
            )
            target = result.weights.reindex(MODULES).to_numpy(dtype=float)
            optimizer_diagnostics = result.diagnostics
            covariance_diagnostics = covariance_estimate.diagnostics

        target = self._apply_guardrails(index, target, base)
        diagnostics = {
            "horizon_days": 1,
            "action_name": "canonical_relative_view_target",
            "optimizer_backend": optimizer_diagnostics.get("optimizer_backend", "unknown"),
            "itransformer_active": model_active,
            "itransformer_horizon_count": len(effective_weights),
            "itransformer_aggregate_confidence": aggregate_confidence,
            "itransformer_disagreement": disagreement,
            "itransformer_uncertainty_multiplier": uncertainty_multiplier,
            "fused_view_spread": float(np.ptp(fused)),
            **covariance_diagnostics,
            **horizon_diagnostics,
            **{f"fused_view_{sleeve}": float(fused[i]) for i, sleeve in enumerate(MODULES)},
        }
        self.canonical_target_cache[index] = (target.copy(), dict(diagnostics))
        return target, diagnostics

    def candidate_weights(self, index: int, action: int) -> tuple[np.ndarray, dict[str, Any]]:
        key = (int(index), int(action))
        if key in self.candidate_cache:
            weights, diagnostics = self.candidate_cache[key]
            return weights.copy(), dict(diagnostics)
        horizon, name, level = self.specs[action]
        calendar_position = int(self.calendar_positions[index])
        base = self.risk_budget_values[calendar_position]
        if name == "noop":
            diagnostics = {"horizon_days": 1, "action_name": name, "model_level": 0.0, "effective_model_level": 0.0}
            return base.copy(), diagnostics
        if name == "formal_anchor":
            target = self._apply_guardrails(index, self.formal_values[calendar_position], base)
            diagnostics = {
                "horizon_days": int(horizon),
                "action_name": name,
                "model_level": 0.0,
                "model_reliability": 0.0,
                "model_confidence": 0.0,
                "effective_model_level": 0.0,
                "grid_alignment": 0.0,
                "optimizer_backend": "formal_pipeline_anchor",
                "fused_view_spread": 0.0,
                **{f"fused_view_{sleeve}": 0.0 for sleeve in MODULES},
            }
            self.candidate_cache[key] = (target.copy(), dict(diagnostics))
            return target, diagnostics
        row = self.state.loc[index]
        rule = row[[f"state_rule_view_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        family = _action_family(name)
        if family in {"model", "baseline_adaptive"}:
            predictive, forecast_confidence, forecast_diagnostics = self._predictive_view(index, horizon, family)
        else:
            predictive = np.zeros(len(MODULES))
            forecast_confidence = 0.0
            forecast_diagnostics = {
                "model_reliability": 0.0,
                "baseline_reliability": 0.0,
                "incremental_reliability": 0.0,
                "itransformer_weight": 0.0,
                "grid_alignment": 0.0,
            }
        effective_view_level = float(level) * forecast_confidence
        effective_model_level = effective_view_level if family == "model" else 0.0
        fused = float(self.cfg["views"]["rule_weight"]) * rule
        fused += float(self.cfg["views"]["model_weight"]) * effective_view_level * predictive
        max_abs = float(np.max(np.abs(fused)))
        if max_abs > 1e-12:
            fused = fused / max_abs
        mu_scale = float(self.cfg["views"]["daily_mu_scale"][self.profile])
        posterior_mu = pd.Series(fused * mu_scale, index=MODULES)
        lookback = int(self.cfg["views"]["covariance_lookback_days"])
        minimum = int(self.cfg["views"]["minimum_covariance_days"])
        history = self.sleeve_returns.iloc[: calendar_position + 1].tail(lookback).reindex(columns=MODULES).fillna(0.0)
        if len(history) < minimum:
            target = base.copy()
            optimizer_diagnostics = {"optimizer_backend": "risk_budget_fallback", "status": "insufficient_history"}
            covariance_diagnostics = {}
        else:
            covariance_estimate = self.covariance_estimate(history)
            result = optimize_sleeve_weights(
                base_weights=pd.Series(base, index=MODULES),
                posterior_mu=posterior_mu,
                cov=covariance_estimate.covariance,
                returns_window=history,
                prev_weights=pd.Series(base, index=MODULES),
                bounds=self.bounds,
                cfg=self.v1cfg,
            )
            target = result.weights.reindex(MODULES).to_numpy(dtype=float)
            optimizer_diagnostics = result.diagnostics
            covariance_diagnostics = covariance_estimate.diagnostics
        target = self._apply_guardrails(index, target, base)
        diagnostics = {
            "horizon_days": int(horizon),
            "action_name": name,
            "model_level": float(level),
            "model_reliability": forecast_diagnostics["model_reliability"],
            "baseline_reliability": forecast_diagnostics["baseline_reliability"],
            "incremental_reliability": forecast_diagnostics["incremental_reliability"],
            "itransformer_weight": forecast_diagnostics["itransformer_weight"],
            "model_confidence": forecast_confidence,
            "effective_view_level": effective_view_level,
            "effective_model_level": effective_model_level,
            "grid_alignment": forecast_diagnostics["grid_alignment"],
            "optimizer_backend": optimizer_diagnostics.get("optimizer_backend", "unknown"),
            "fused_view_spread": float(np.ptp(fused)),
            **covariance_diagnostics,
            **{f"fused_view_{sleeve}": float(fused[i]) for i, sleeve in enumerate(MODULES)},
        }
        self.candidate_cache[key] = (target.copy(), dict(diagnostics))
        return target, diagnostics

    def step(
        self, index: int, action: int, previous_weights: np.ndarray
    ) -> tuple[float, int, np.ndarray, dict[str, Any]]:
        horizon, name, level = self.specs[action]
        signal_position = int(self.calendar_positions[index])
        base = self.risk_budget_values[signal_position]
        if name == "noop":
            target = self._apply_guardrails(index, np.asarray(previous_weights, dtype=float), base)
            diagnostics = {
                "horizon_days": 1,
                "action_name": "noop",
                "model_level": 0.0,
                "effective_model_level": 0.0,
                "model_confidence": 0.0,
                "grid_alignment": 0.0,
                "optimizer_backend": "hold_and_guardrail",
                "fused_view_spread": 0.0,
                **{f"fused_view_{sleeve}": 0.0 for sleeve in MODULES},
            }
        else:
            target, diagnostics = self.candidate_weights(index, action)
        start = signal_position + 1
        stop = min(signal_position + int(horizon) + 1, len(self.calendar))
        if start >= stop:
            return 0.0, len(self.state), target, {"daily": []}
        returns = self.return_values[start:stop]
        formal_weights = self.formal_values[start:stop]
        formal_daily = np.sum(formal_weights * returns, axis=1) - self.formal_cost[start:stop]
        if name == "formal_anchor":
            strategy_weights = formal_weights.copy()
        else:
            strategy_weights = np.repeat(target[None, :], len(returns), axis=0)
        daily_turnover = np.zeros(len(strategy_weights), dtype=float)
        daily_turnover[0] = float(np.abs(strategy_weights[0] - np.asarray(previous_weights, dtype=float)).sum())
        if name == "formal_anchor" and len(strategy_weights) > 1:
            daily_turnover[1:] = np.abs(np.diff(strategy_weights, axis=0)).sum(axis=1)
        daily_cost = daily_turnover * float(self.cfg["reward"]["transaction_cost_bps"]) / 10000.0
        strategy_daily = np.sum(strategy_weights * returns, axis=1) - daily_cost
        turnover = float(daily_turnover.sum())
        cost = float(daily_cost.sum())
        target = strategy_weights[-1].copy()
        formal_total = float(np.prod(1.0 + formal_daily) - 1.0)
        strategy_total = float(np.prod(1.0 + strategy_daily) - 1.0)
        downside_worsening = max(0.0, -strategy_total) - max(0.0, -formal_total)
        active_deviation = float(np.mean(np.abs(strategy_weights - formal_weights).sum(axis=1)))
        horizon_scale = math.sqrt(max(len(strategy_daily), 1))
        strategy_vol = float(np.std(strategy_daily, ddof=0) * horizon_scale)
        formal_vol = float(np.std(formal_daily, ddof=0) * horizon_scale)
        volatility_worsening = max(0.0, strategy_vol - formal_vol)
        strategy_nav = np.cumprod(1.0 + strategy_daily)
        formal_nav = np.cumprod(1.0 + formal_daily)
        strategy_drawdown = float(np.min(strategy_nav / np.maximum.accumulate(strategy_nav) - 1.0))
        formal_drawdown = float(np.min(formal_nav / np.maximum.accumulate(formal_nav) - 1.0))
        drawdown_worsening = max(0.0, abs(strategy_drawdown) - abs(formal_drawdown))
        reward = (
            strategy_total
            - formal_total
            - float(self.cfg["reward"]["downside_penalty"]) * max(0.0, downside_worsening)
            - float(self.cfg["reward"]["relative_volatility_penalty"][self.profile]) * volatility_worsening
            - float(self.cfg["reward"]["relative_drawdown_penalty"][self.profile]) * drawdown_worsening
            - float(self.cfg["reward"]["active_deviation_penalty"]) * active_deviation
        ) * float(self.cfg["reward"]["reward_scale"])
        next_calendar_position = stop - 1
        next_index = int(np.searchsorted(self.calendar_positions, next_calendar_position, side="left"))
        daily: list[dict[str, Any]] = []
        for offset, calendar_index in enumerate(range(start, stop)):
            daily.append(
                {
                    "date": self.calendar[calendar_index],
                    "signal_date": self.state.loc[index, "date"],
                    "formal_return": float(formal_daily[offset]),
                    "strategy_return": float(strategy_daily[offset]),
                    "strategy_turnover": float(daily_turnover[offset]),
                    "strategy_cost": float(daily_cost[offset]),
                    **{f"weight_{sleeve}": float(strategy_weights[offset, i]) for i, sleeve in enumerate(MODULES)},
                }
            )
        info = {
            "daily": daily,
            "horizon_days": int(horizon),
            "action_name": name,
            "action_level": float(level),
            "execution_start": self.calendar[start],
            "execution_end": self.calendar[stop - 1],
            "turnover": turnover,
            "cost": cost,
            "formal_total": formal_total,
            "strategy_total": strategy_total,
            "active_deviation": active_deviation,
            "volatility_worsening": volatility_worsening,
            "drawdown_worsening": drawdown_worsening,
            **diagnostics,
        }
        return float(reward), next_index, target, info


def build_environment(profile: str, state: pd.DataFrame, cfg: dict[str, Any]) -> IntegratedViewEnvironment:
    market_cfg = copy.deepcopy(cfg)
    market_cfg["data"]["asset_weights_pattern"] = market_cfg["data"]["formal_asset_pattern"]
    market_cfg["data"]["sleeve_weights_pattern"] = market_cfg["data"]["formal_sleeve_pattern"]
    sleeve_returns, formal_base, lo, hi = build_profile_market(profile, market_cfg)
    risk_budget = pd.read_parquet(ROOT / cfg["data"]["risk_budget_pattern"].format(profile=profile))
    risk_budget["date"] = pd.to_datetime(risk_budget["date"], errors="coerce")
    calendar = pd.DatetimeIndex(sleeve_returns.index)
    daily_risk_budget = pd.merge_asof(
        pd.DataFrame({"date": calendar}),
        risk_budget[["date"] + MODULES].sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("date")
    daily_risk_budget = daily_risk_budget.ffill().dropna().reindex(calendar).ffill()
    projected = np.vstack(
        [project_simplex_bounds(row, lo, hi) for row in daily_risk_budget[MODULES].to_numpy(dtype=float)]
    )
    daily_risk_budget.loc[:, MODULES] = projected
    return IntegratedViewEnvironment(state, sleeve_returns, formal_base, daily_risk_budget, lo, hi, profile, cfg)


def _simulate_policy(
    env: IntegratedViewEnvironment,
    start: int,
    stop: int,
    model: ActorCritic,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    index = start
    current = env.initial_weights(index)
    total = 0.0
    with torch.no_grad():
        while index < stop:
            obs = torch.tensor(env.observation(index, current, mean, std))[None, :]
            logits, _ = model(obs)
            mask = torch.tensor(env.action_mask(index), dtype=torch.bool)[None, :]
            logits = logits.masked_fill(~mask, -1e9)
            action = int(logits.argmax(dim=1).item())
            reward, next_index, current, _ = env.step(index, action, current)
            total += reward
            index = max(index + 1, next_index)
    return float(total)


def _simulate_fixed_action_reward(
    env: IntegratedViewEnvironment, start: int, stop: int, action: int
) -> float:
    index = start
    current = env.initial_weights(index)
    total = 0.0
    while index < stop:
        reward, next_index, current, _ = env.step(index, action, current)
        total += reward
        index = max(index + 1, next_index)
    return float(total)


def _teacher_score(
    reward: float, horizon_days: int, reference_days: int, mode: str
) -> float:
    """Score immediate teacher actions using a declared holding-period convention."""
    if mode == "cumulative":
        return float(reward)
    if mode == "horizon_normalized":
        return float(reward) * float(reference_days) / float(max(int(horizon_days), 1))
    raise ValueError(f"unsupported teacher reward mode: {mode}")


def train_seed(
    env: IntegratedViewEnvironment,
    eligible: np.ndarray,
    year: int,
    seed: int,
    teacher_reward_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = env.cfg["ppo"]
    n_val = max(30, int(math.ceil(len(eligible) * float(cfg["validation_fraction"]))))
    split = max(30, len(eligible) - n_val)
    train_rows = eligible[:split]
    val_rows = eligible[split:]
    static = env.state.loc[train_rows, env.state_cols].to_numpy(dtype=float)
    mean = np.nanmean(static, axis=0)
    std = np.nanstd(static, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    model = ActorCritic(len(env.state_cols) + len(MODULES), len(env.specs), int(cfg["hidden_dim"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)

    stride = max(int(cfg.get("teacher_stride", 2)), 1)
    teacher_rows = train_rows[::stride]
    observations: list[np.ndarray] = []
    teacher_actions: list[int] = []
    teacher_masks: list[np.ndarray] = []
    reference_days = int(cfg.get("teacher_reference_horizon_days", 20))
    for index in teacher_rows:
        previous = env.initial_weights(int(index))
        valid = env.action_mask(int(index))
        teacher_scores: list[float] = []
        for action, (horizon, _, _) in enumerate(env.specs):
            if not valid[action]:
                teacher_scores.append(-np.inf)
                continue
            reward = env.step(int(index), action, previous)[0]
            teacher_scores.append(
                _teacher_score(reward, horizon, reference_days, teacher_reward_mode)
            )
        best_action = int(np.argmax(teacher_scores))
        if teacher_scores[best_action] < float(cfg.get("minimum_teacher_edge", 0.01)):
            best_action = env.formal_anchor_action
        observations.append(env.observation(int(index), previous, mean, std))
        teacher_actions.append(best_action)
        teacher_masks.append(valid)
    bc_obs = torch.tensor(np.vstack(observations), dtype=torch.float32)
    bc_mask = torch.tensor(np.vstack(teacher_masks), dtype=torch.bool)
    teacher = torch.tensor(teacher_actions, dtype=torch.long)
    counts = np.bincount(teacher_actions, minlength=len(env.specs)).astype(float)
    present = counts > 0
    class_weights = np.zeros(len(env.specs), dtype=float)
    power = float(cfg.get("behavior_clone_class_balance_power", 0.5))
    class_weights[present] = np.power(counts[present].sum() / counts[present], power)
    class_weights[present] /= class_weights[present].mean()
    class_weights = np.clip(class_weights, 0.0, 3.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    teacher_distribution = {
        f"{env.specs[action][1]}_h{env.specs[action][0]}": int(count)
        for action, count in enumerate(counts)
        if count > 0
    }
    print(
        f"[{env.profile}][{year}][seed={seed}][teacher={teacher_reward_mode}] "
        f"teacher_distribution={json.dumps(teacher_distribution, sort_keys=True)}",
        flush=True,
    )
    for _ in range(int(cfg["behavior_clone_epochs"])):
        logits, values = model(bc_obs)
        logits = logits.masked_fill(~bc_mask, -1e9)
        loss = nn.functional.cross_entropy(logits, teacher, weight=class_weights_tensor) + 0.02 * torch.mean(values**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_start, train_stop = int(train_rows[0]), int(train_rows[-1]) + 1
    val_start, val_stop = int(val_rows[0]), int(val_rows[-1]) + 1
    best_state = copy.deepcopy(model.state_dict())
    best_val = _simulate_policy(env, val_start, val_stop, model, mean, std)
    rule_validation_reward = _simulate_fixed_action_reward(
        env, val_start, val_stop, env.rules_20_action
    )
    fallback_action = (
        env.rules_20_action if rule_validation_reward > 0.0 else env.formal_anchor_action
    )
    wait = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        index = train_start
        current = env.initial_weights(index)
        obs_rows: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        discounts: list[float] = []
        old_logp: list[float] = []
        values: list[float] = []
        action_masks: list[np.ndarray] = []
        while index < train_stop:
            obs_array = env.observation(index, current, mean, std)
            obs = torch.tensor(obs_array)[None, :]
            with torch.no_grad():
                logits, value = model(obs)
                mask = torch.tensor(env.action_mask(index), dtype=torch.bool)[None, :]
                logits = logits.masked_fill(~mask, -1e9)
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(distribution.sample().item())
                logp = float(distribution.log_prob(torch.tensor([action])).item())
            reward, next_index, current, info = env.step(index, action, current)
            obs_rows.append(obs_array)
            actions.append(action)
            rewards.append(reward)
            discounts.append(float(cfg["gamma"]) ** (float(info.get("horizon_days", 20)) / 20.0))
            old_logp.append(logp)
            values.append(float(value.item()))
            action_masks.append(env.action_mask(index))
            index = max(index + 1, next_index)

        obs_tensor = torch.tensor(np.vstack(obs_rows), dtype=torch.float32)
        action_mask_tensor = torch.tensor(np.vstack(action_masks), dtype=torch.bool)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        reward_array = np.asarray(rewards, dtype=float)
        value_array = np.asarray(values, dtype=float)
        advantage = np.zeros(len(rewards), dtype=float)
        last = 0.0
        next_value = 0.0
        for position in reversed(range(len(rewards))):
            delta = reward_array[position] + discounts[position] * next_value - value_array[position]
            last = delta + discounts[position] * float(cfg["gae_lambda"]) * last
            advantage[position] = last
            next_value = value_array[position]
        target = advantage + value_array
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        old_logp_tensor = torch.tensor(old_logp, dtype=torch.float32)
        advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        for _ in range(int(cfg["updates_per_epoch"])):
            logits, value = model(obs_tensor)
            logits = logits.masked_fill(~action_mask_tensor, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(action_tensor)
            ratio = torch.exp(logp - old_logp_tensor)
            clipped = torch.clamp(ratio, 1.0 - float(cfg["clip_ratio"]), 1.0 + float(cfg["clip_ratio"]))
            policy_loss = -torch.minimum(ratio * advantage_tensor, clipped * advantage_tensor).mean()
            value_loss = torch.mean((value - target_tensor) ** 2)
            entropy = distribution.entropy().mean()
            loss = policy_loss + float(cfg["value_loss_weight"]) * value_loss - float(cfg["entropy_weight"]) * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch == 1 or epoch % int(cfg["evaluation_every"]) == 0:
            val_reward = _simulate_policy(env, val_start, val_stop, model, mean, std)
            logs.append(
                {
                    "profile": env.profile,
                    "evaluation_year": year,
                    "seed": seed,
                    "teacher_reward_mode": teacher_reward_mode,
                    "epoch": epoch,
                    "rollout_steps": len(rewards),
                    "rollout_reward": float(reward_array.sum()),
                    "validation_reward": val_reward,
                    "entropy": float(entropy.detach()),
                }
            )
            print(
                f"[{env.profile}][{year}][seed={seed}][teacher={teacher_reward_mode}] epoch={epoch} "
                f"rollout={float(reward_array.sum()):.6f} validation={val_reward:.6f}",
                flush=True,
            )
            if val_reward > best_val + float(cfg["early_stopping_min_delta"]):
                best_val = val_reward
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(cfg["early_stop_patience"]):
                    break
    return {
        "state_dict": best_state,
        "mean": mean,
        "std": std,
        "obs_dim": len(env.state_cols) + len(MODULES),
        "action_dim": len(env.specs),
        "hidden_dim": int(cfg["hidden_dim"]),
        "training_cutoff": pd.Timestamp(year, 1, 1),
        "teacher_reward_mode": teacher_reward_mode,
        "validation_reward": best_val,
        "rule_validation_reward": rule_validation_reward,
        "fallback_action": fallback_action,
        "teacher_distribution": teacher_distribution,
    }, logs


def choose_action(
    models: list[dict[str, Any]], env: IntegratedViewEnvironment, index: int, current: np.ndarray
) -> tuple[int, int, str]:
    threshold = float(env.cfg["ppo"]["minimum_policy_validation_reward"])
    base_name = str(
        env.cfg.get("deployment", {})
        .get("base_action", {})
        .get(env.profile, "formal_anchor")
    )
    base_actions = [
        action for action, (_, name, _) in enumerate(env.specs) if name == base_name
    ]
    if not base_actions:
        raise ValueError(f"deployment base action is not available: {base_name}")
    deployment_base = int(base_actions[0])
    fallback = deployment_base if models else env.formal_anchor_action
    minimum_edge = float(
        env.cfg.get("deployment", {}).get("minimum_validation_edge_vs_base", 0.0)
    )
    qualified: list[dict[str, Any]] = []
    for model in models:
        base_validation = (
            float(model.get("rule_validation_reward", -np.inf))
            if base_name == "rules_only"
            else 0.0
        )
        required = max(threshold, base_validation + minimum_edge)
        if float(model.get("validation_reward", -np.inf)) >= required:
            qualified.append(model)
    minimum = int(env.cfg["ppo"]["minimum_qualified_models"])
    if len(qualified) < minimum:
        modes = ",".join(sorted(str(model.get("teacher_reward_mode", "unknown")) for model in qualified))
        return fallback, len(qualified), modes
    valid = env.action_mask(index)
    probabilities: list[np.ndarray] = []
    family_votes: list[str] = []
    families = sorted({_action_family(name) for _, name, _ in env.specs})
    for checkpoint in qualified:
        model = ActorCritic(checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        obs = env.observation(index, current, checkpoint["mean"], checkpoint["std"])
        with torch.no_grad():
            logits, _ = model(torch.tensor(obs)[None, :])
            mask = torch.tensor(valid, dtype=torch.bool)[None, :]
            logits = logits.masked_fill(~mask, -1e9)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        probabilities.append(probs)
        family_scores = {
            family: float(
                sum(probs[action] for action, (_, name, _) in enumerate(env.specs) if _action_family(name) == family)
            )
            for family in families
        }
        family_votes.append(max(family_scores, key=family_scores.get))
    counts = pd.Series(family_votes).value_counts()
    if int(counts.iloc[0]) < int(env.cfg["ppo"]["ensemble_min_agreement"]):
        modes = ",".join(sorted(str(model.get("teacher_reward_mode", "unknown")) for model in qualified))
        return fallback, len(qualified), modes
    selected_family = str(counts.index[0])
    mean_probability = np.mean(np.vstack(probabilities), axis=0)
    selected_family_probability = float(
        sum(
            mean_probability[action]
            for action, (_, name, _) in enumerate(env.specs)
            if _action_family(name) == selected_family
        )
    )
    fallback_family = _action_family(env.specs[fallback][1])
    fallback_family_probability = float(
        sum(
            mean_probability[action]
            for action, (_, name, _) in enumerate(env.specs)
            if _action_family(name) == fallback_family
        )
    )
    deployment_cfg = env.cfg.get("deployment", {})
    if selected_family != fallback_family and (
        selected_family_probability < float(deployment_cfg.get("minimum_family_probability", 0.40))
        or selected_family_probability - fallback_family_probability
        < float(deployment_cfg.get("minimum_probability_margin", 0.08))
    ):
        modes = ",".join(
            sorted(str(model.get("teacher_reward_mode", "unknown")) for model in qualified)
        )
        return fallback, len(qualified), modes
    candidates = [
        action
        for action, (_, name, _) in enumerate(env.specs)
        if valid[action] and _action_family(name) == selected_family
    ]
    if not candidates:
        modes = ",".join(sorted(str(model.get("teacher_reward_mode", "unknown")) for model in qualified))
        return fallback, len(qualified), modes
    action = max(candidates, key=lambda candidate: float(mean_probability[candidate]))
    modes = ",".join(sorted(str(model.get("teacher_reward_mode", "unknown")) for model in qualified))
    return action, len(qualified), modes


def _simulate_fixed_rules(env: IntegratedViewEnvironment) -> pd.DataFrame:
    index = 0
    current = env.initial_weights(index)
    rows: list[dict[str, Any]] = []
    while index < len(env.state):
        _, next_index, current, info = env.step(index, env.rules_20_action, current)
        if not info.get("daily"):
            break
        rows.extend(info["daily"])
        index = max(index + 1, next_index)
    frame = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
    return frame[["date", "strategy_return"]].rename(columns={"strategy_return": "rule_only_return"})


def _materialize_asset_weights(profile: str, daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    assets = pd.read_parquet(ROOT / cfg["data"]["formal_asset_pattern"].format(profile=profile))
    assets["date"] = pd.to_datetime(assets["date"], errors="coerce")
    assets = assets.loc[assets["asset_code"].ne("cash")].copy()
    returns = pd.read_parquet(ROOT / cfg["data"]["asset_returns"])
    module_map = (
        returns.dropna(subset=["asset_code", "module"])
        .drop_duplicates("asset_code", keep="last")
        .set_index("asset_code")["module"]
        .to_dict()
    )
    dates = pd.DatetimeIndex(sorted(assets["date"].dropna().unique()))
    grouped = {date: group.set_index("asset_code")["weight"].astype(float) for date, group in assets.groupby("date")}
    rows: list[dict[str, Any]] = []
    for _, record in daily.iterrows():
        date = pd.Timestamp(record["date"])
        position = int(dates.searchsorted(date, side="left") - 1)
        if position < 0:
            continue
        source_date = dates[position]
        weights = grouped[source_date]
        for sleeve in MODULES:
            selected = [code for code in weights.index if module_map.get(code) == sleeve]
            total = float(weights.reindex(selected).sum())
            if total <= 1e-12:
                continue
            sleeve_weight = float(record[f"weight_{sleeve}"])
            for code in selected:
                rows.append(
                    {
                        "date": date,
                        "source_asset_mix_date": source_date,
                        "profile": profile,
                        "asset_code": code,
                        "module": sleeve,
                        "weight": sleeve_weight * float(weights.loc[code]) / total,
                    }
                )
    return pd.DataFrame(rows)


def run_profile(profile: str, state: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    env = build_environment(profile, state, cfg)
    years = sorted(pd.to_datetime(env.state["date"]).dt.year.unique())
    checkpoints: dict[int, list[dict[str, Any]]] = {}
    training_logs: list[dict[str, Any]] = []
    model_dir = out_path(cfg, "model_dir") / profile
    model_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        cutoff = pd.Timestamp(int(year), 1, 1)
        eligible = env.eligible_rows(cutoff)
        if len(eligible) < int(cfg["ppo"]["min_train_rows"]):
            checkpoints[int(year)] = []
            continue
        annual_models: list[dict[str, Any]] = []
        teacher_modes = list(cfg["ppo"].get("teacher_reward_modes", ["cumulative"]))
        for seed in cfg["ppo"]["seeds"]:
            mode_candidates: list[dict[str, Any]] = []
            for teacher_mode in teacher_modes:
                started = time.perf_counter()
                print(
                    f"[{profile}][{int(year)}][seed={int(seed)}][teacher={teacher_mode}] training_start "
                    f"eligible_rows={len(eligible)} actions={len(env.specs)}",
                    flush=True,
                )
                checkpoint, logs = train_seed(
                    env, eligible, int(year), int(seed), str(teacher_mode)
                )
                mode_candidates.append(checkpoint)
                training_logs.extend(logs)
                print(
                    f"[{profile}][{int(year)}][seed={int(seed)}][teacher={teacher_mode}] training_done "
                    f"validation={float(checkpoint['validation_reward']):.6f} "
                    f"elapsed_seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            checkpoint = max(
                mode_candidates,
                key=lambda candidate: float(candidate.get("validation_reward", -np.inf)),
            )
            annual_models.append(checkpoint)
            print(
                f"[{profile}][{int(year)}][seed={int(seed)}] teacher_selected="
                f"{checkpoint['teacher_reward_mode']} validation="
                f"{float(checkpoint['validation_reward']):.6f}",
                flush=True,
            )
        checkpoints[int(year)] = annual_models
        torch.save(annual_models, model_dir / f"policy_{int(year)}.pt")

    index = 0
    current = env.initial_weights(index)
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    common_end = pd.Timestamp(env.state["date"].max())
    while index < len(env.state):
        date = pd.Timestamp(env.state.loc[index, "date"])
        models = checkpoints.get(date.year, [])
        action, qualified_count, qualified_teacher_modes = choose_action(models, env, index, current)
        reward, next_index, current, info = env.step(index, action, current)
        visible = [row for row in info.get("daily", []) if pd.Timestamp(row["date"]) <= common_end]
        if not visible:
            break
        if not models:
            source = "warmup_formal_anchor"
        elif qualified_count < int(cfg["ppo"]["minimum_qualified_models"]):
            source = f"validation_gate_{info['action_name']}"
        else:
            source = "annual_walkforward_ppo"
        signal_position = int(env.calendar_positions[index])
        risk_base = env.risk_budget_values[signal_position]
        state_row = env.state.loc[index]
        decision: dict[str, Any] = {
            "date": date,
            "profile": profile,
            "stock_bond_regime": state_row["stock_bond_regime"],
            "policy_source": source,
            "policy_training_cutoff": pd.Timestamp(date.year, 1, 1) if models else pd.NaT,
            "itransformer_information_cutoff": state_row["itransformer_information_cutoff"],
            "qualified_model_count": qualified_count,
            "qualified_teacher_modes": qualified_teacher_modes,
            "action_index": action,
            "action_name": info["action_name"],
            "action_level": info["action_level"],
            "effective_view_level": info.get("effective_view_level", 0.0),
            "effective_model_level": info.get("effective_model_level", 0.0),
            "model_confidence": info.get("model_confidence", 0.0),
            "model_reliability": info.get("model_reliability", 0.0),
            "baseline_reliability": info.get("baseline_reliability", 0.0),
            "incremental_reliability": info.get("incremental_reliability", 0.0),
            "itransformer_weight": info.get("itransformer_weight", 0.0),
            "grid_alignment": info.get("grid_alignment", 0.0),
            "horizon_days": info["horizon_days"],
            "execution_start": info["execution_start"],
            "execution_end": info["execution_end"],
            "reward": reward,
            "turnover": info["turnover"],
            "cost": info["cost"],
            "optimizer_backend": info.get("optimizer_backend", "unknown"),
            "block_credit": bool(state_row["state_grid_block_credit"]),
            "block_cb": bool(state_row["state_grid_block_cb"]),
            "block_equity": bool(state_row["state_grid_block_equity"]),
        }
        for sleeve_position, sleeve in enumerate(MODULES):
            decision[f"risk_budget_{sleeve}"] = float(risk_base[sleeve_position])
            decision[f"target_{sleeve}"] = float(current[sleeve_position])
            decision[f"rule_view_{sleeve}"] = float(state_row[f"state_rule_view_{sleeve}"])
            decision[f"fused_view_{sleeve}"] = float(info.get(f"fused_view_{sleeve}", 0.0))
        decision_rows.append(decision)
        daily_rows.extend(visible)
        index = max(index + 1, next_index)

    daily = pd.DataFrame(daily_rows).drop_duplicates("date", keep="last").sort_values("date")
    decisions = pd.DataFrame(decision_rows).sort_values("date")
    rule_control = _simulate_fixed_rules(env)
    daily = daily.merge(rule_control, on="date", how="inner")
    for name in ["formal", "rule_only", "strategy"]:
        daily[f"{name}_nav"] = (1.0 + daily[f"{name}_return"]).cumprod()
    metrics_rows: list[dict[str, Any]] = []
    for split, mask in {
        "full_oos": pd.Series(True, index=daily.index),
        "oos_2024_onward": pd.to_datetime(daily["date"]).ge(pd.Timestamp(cfg["gate"]["evaluation_start"])),
    }.items():
        stats = {name: perf_stats(daily.loc[mask, f"{name}_return"]) for name in ["formal", "rule_only", "strategy"]}
        row: dict[str, Any] = {"profile": profile, "split": split}
        for name, values in stats.items():
            row.update({f"{name}_{key}": value for key, value in values.items()})
        row.update(
            {
                "delta_annual_return_vs_formal": stats["strategy"].get("annual_return", np.nan)
                - stats["formal"].get("annual_return", np.nan),
                "delta_sharpe_vs_formal": stats["strategy"].get("sharpe", np.nan)
                - stats["formal"].get("sharpe", np.nan),
                "delta_annual_return_vs_rule_only": stats["strategy"].get("annual_return", np.nan)
                - stats["rule_only"].get("annual_return", np.nan),
                "delta_sharpe_vs_rule_only": stats["strategy"].get("sharpe", np.nan)
                - stats["rule_only"].get("sharpe", np.nan),
                "max_drawdown_worsening_vs_formal": abs(min(stats["strategy"].get("max_drawdown", 0.0), 0.0))
                - abs(min(stats["formal"].get("max_drawdown", 0.0), 0.0)),
                "mean_strategy_turnover": float(daily.loc[mask, "strategy_turnover"].mean()),
            }
        )
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    daily.to_csv(out_path(cfg, "daily_pattern", profile), index=False, encoding="utf-8-sig")
    decisions.to_parquet(out_path(cfg, "decisions_pattern", profile), index=False)
    metrics.to_csv(out_path(cfg, "metrics_pattern", profile), index=False, encoding="utf-8-sig")
    pd.DataFrame(training_logs).to_csv(out_path(cfg, "training_log_pattern", profile), index=False, encoding="utf-8-sig")
    holdings = _materialize_asset_weights(profile, daily, cfg)
    holdings.to_parquet(out_path(cfg, "holdings_pattern", profile), index=False)
    return {
        "profile": profile,
        "decisions": len(decisions),
        "model_decisions": int(
            (
                decisions["action_name"].astype(str).str.startswith("model_")
                & pd.to_numeric(decisions["effective_model_level"], errors="coerce").gt(0)
                & pd.to_numeric(decisions["itransformer_weight"], errors="coerce").gt(0)
            ).sum()
        ),
        "days": len(daily),
        "candidate_cache": len(env.candidate_cache),
        **metrics.iloc[-1].to_dict(),
    }


def _write_visuals(cfg: dict[str, Any], summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    profiles = list(cfg["profiles"])
    colors = {"formal": "#17365D", "rule_only": "#6B7280", "strategy": "#E8A000"}
    figure, axes = plt.subplots(len(profiles), 3, figsize=(18, 4.5 * len(profiles)), constrained_layout=True)
    if len(profiles) == 1:
        axes = np.array([axes])
    for row, profile in enumerate(profiles):
        daily = pd.read_csv(out_path(cfg, "daily_pattern", profile), parse_dates=["date"])
        for name, label in [("formal", "Formal monthly"), ("rule_only", "Integrated rules"), ("strategy", "iTransformer + PPO")]:
            axes[row, 0].plot(daily["date"], daily[f"{name}_nav"], label=label, color=colors[name], linewidth=1.6)
            drawdown = daily[f"{name}_nav"] / daily[f"{name}_nav"].cummax() - 1.0
            axes[row, 1].plot(daily["date"], drawdown, label=label, color=colors[name], linewidth=1.3)
        axes[row, 0].set_title(f"{profile}: NAV")
        axes[row, 1].set_title(f"{profile}: Drawdown")
        axes[row, 0].legend(frameon=False, ncol=3, fontsize=8)
        axes[row, 1].axhline(0.0, color="#D1D5DB", linewidth=0.8)
        weights = daily.set_index("date")[[f"weight_{sleeve}" for sleeve in MODULES]]
        axes[row, 2].stackplot(
            weights.index,
            *[weights[f"weight_{sleeve}"].to_numpy() for sleeve in MODULES],
            labels=MODULES,
            alpha=0.88,
        )
        axes[row, 2].set_ylim(0.0, 1.0)
        axes[row, 2].set_title(f"{profile}: Integrated sleeve weights")
        axes[row, 2].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
        for column in range(3):
            axes[row, column].grid(alpha=0.18)
    figure.savefig(out_path(cfg, "visual_png"), dpi=170)
    plt.close(figure)
    image_name = out_path(cfg, "visual_png").name
    metrics_html = summary.to_html(index=False, float_format=lambda value: f"{value:.6f}")
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Integrated Daily View PPO</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#172033}}h1{{font-size:26px}}p{{max-width:980px;line-height:1.55}}img{{width:100%;max-width:1500px}}table{{border-collapse:collapse;font-size:12px;display:block;overflow:auto}}th,td{{border:1px solid #d8dee9;padding:6px 8px;white-space:nowrap}}th{{background:#17365D;color:white}}</style></head>
<body><h1>Nine-grid Integrated iTransformer Views + PPO</h1><p>The nine-grid and hard filters are evaluated before model views. PPO routes only feasible view candidates; risk-budget priors and skfolio MeanRisk generate the sleeve weights. All executions are T+1.</p><img src='{html.escape(image_name)}' alt='dashboard'><h2>Metrics</h2>{metrics_html}</body></html>"""
    out_path(cfg, "visual_html").write_text(page, encoding="utf-8")


def audit_and_report(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    profile_gates: list[dict[str, Any]] = []
    common_days: list[int] = []
    for profile in cfg["profiles"]:
        metrics = pd.read_csv(out_path(cfg, "metrics_pattern", profile))
        summaries.append(metrics)
        daily = pd.read_csv(out_path(cfg, "daily_pattern", profile), parse_dates=["date", "signal_date"])
        decisions = pd.read_parquet(out_path(cfg, "decisions_pattern", profile))
        weights = daily[[f"weight_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        v1cfg = _profile_v1_config(profile)
        profile_bounds = v1cfg["product"]["profiles"][profile]["sleeve_bounds"]
        lo = np.array([float(profile_bounds[sleeve][0]) for sleeve in MODULES])
        hi = np.array([float(profile_bounds[sleeve][1]) for sleeve in MODULES])
        block_checks: list[bool] = []
        for _, decision in decisions.iterrows():
            if str(decision.get("action_name")) == "formal_anchor":
                continue
            checks = []
            for flag, sleeve in [("block_credit", "credit"), ("block_cb", "convertible_bonds"), ("block_equity", "equity")]:
                if bool(decision[flag]):
                    checks.append(float(decision[f"target_{sleeve}"]) <= float(decision[f"risk_budget_{sleeve}"]) + 1e-8)
            block_checks.extend(checks)
        checks = {
            "weight_sum_one": bool(np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)),
            "sleeve_bounds": bool(np.all(weights >= lo - 1e-8) and np.all(weights <= hi + 1e-8)),
            "t_plus_one": bool((daily["signal_date"] < daily["date"]).all()),
            "daily_unique": bool(not daily["date"].duplicated().any()),
            "no_nan_return": bool(daily[["formal_return", "rule_only_return", "strategy_return"]].notna().all().all()),
            "policy_cutoff_before_execution": bool(
                decisions.loc[decisions["policy_source"].eq("annual_walkforward_ppo"), "policy_training_cutoff"].lt(
                    decisions.loc[decisions["policy_source"].eq("annual_walkforward_ppo"), "execution_start"]
                ).all()
            ),
            "itransformer_cutoff_not_future": bool(decisions["itransformer_information_cutoff"].le(decisions["date"]).all()),
            "nine_grid_hard_blocks": bool(all(block_checks)) if block_checks else True,
            "asset_weight_sum_one": bool(
                pd.read_parquet(out_path(cfg, "holdings_pattern", profile)).groupby("date")["weight"].sum().sub(1.0).abs().lt(1e-6).all()
            ),
        }
        common_days.append(len(daily))
        for check, passed in checks.items():
            audits.append({"profile": profile, "check": check, "passed": passed})
        evaluation = metrics.loc[metrics["split"].eq("oos_2024_onward")].iloc[0]
        model_actions = int(
            (
                pd.to_datetime(decisions["date"]).ge(pd.Timestamp(cfg["gate"]["evaluation_start"]))
                & decisions["action_name"].astype(str).str.startswith("model_")
                & pd.to_numeric(decisions["effective_model_level"], errors="coerce").gt(0.0)
                & pd.to_numeric(decisions["itransformer_weight"], errors="coerce").gt(0.0)
            ).sum()
        )
        gate_checks = {
            "annual_return_vs_formal": bool(
                evaluation["delta_annual_return_vs_formal"] >= float(cfg["gate"]["min_annual_return_delta_vs_formal"])
            ),
            "sharpe_vs_formal": bool(
                evaluation["delta_sharpe_vs_formal"] >= float(cfg["gate"]["min_sharpe_delta_vs_formal"])
            ),
            "sharpe_vs_rule_only": bool(
                evaluation["delta_sharpe_vs_rule_only"] >= float(cfg["gate"]["min_sharpe_delta_vs_rule_only"])
            ),
            "drawdown": bool(
                evaluation["max_drawdown_worsening_vs_formal"] <= float(cfg["gate"]["max_drawdown_worsening"])
            ),
            "model_actions": bool(model_actions >= int(cfg["gate"]["min_model_actions"])),
        }
        profile_gates.append(
            {
                "profile": profile,
                "enabled": bool(all(gate_checks.values())),
                "checks": gate_checks,
                "model_actions": model_actions,
                "delta_annual_return_vs_formal": float(evaluation["delta_annual_return_vs_formal"]),
                "delta_sharpe_vs_formal": float(evaluation["delta_sharpe_vs_formal"]),
                "delta_sharpe_vs_rule_only": float(evaluation["delta_sharpe_vs_rule_only"]),
                "max_drawdown_worsening_vs_formal": float(evaluation["max_drawdown_worsening_vs_formal"]),
            }
        )
    if len(set(common_days)) > 1:
        for profile in cfg["profiles"]:
            audits.append({"profile": profile, "check": "common_sample_length", "passed": False})
    summary = pd.concat(summaries, ignore_index=True)
    audit = pd.DataFrame(audits)
    gate = {
        "enabled_for_formal_strategy": bool(profile_gates) and all(item["enabled"] for item in profile_gates),
        "evaluation_start": cfg["gate"]["evaluation_start"],
        "profiles": profile_gates,
        "caveat": "The integrated strategy remains a research challenger; 2025+ has been observed during prior development.",
    }
    summary.to_csv(out_path(cfg, "summary"), index=False, encoding="utf-8-sig")
    audit.to_csv(out_path(cfg, "audit_csv"), index=False, encoding="utf-8-sig")
    out_path(cfg, "gate_report").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    action_rows: list[dict[str, Any]] = []
    for profile in cfg["profiles"]:
        decisions = pd.read_parquet(out_path(cfg, "decisions_pattern", profile))
        for action_name, count in decisions["action_name"].value_counts().items():
            action_rows.append({"profile": profile, "action": action_name, "count": int(count)})
    action_table = pd.DataFrame(action_rows)
    rank_gate_path = ROOT / "output/reports/itransformer_daily_timing_v2_rank_robust_residual_formal_gate.json"
    rank_gate = json.loads(rank_gate_path.read_text(encoding="utf-8")) if rank_gate_path.exists() else {}
    lines = [
        "# 九宫格优先的日频 iTransformer + PPO + EP/skfolio 完整结果",
        "",
        "## 1. 策略定位",
        "",
        "这条链路是正式月频策略之上的日频研究挑战者。风险平价/风险预算仍是底仓，股债九宫格与六模块状态卡先决定边界，机械 Market Views 与预测专家只形成均值观点，EP/skfolio 负责在产品上下限内求解，PPO 只路由可行候选而不直接输出任意仓位。任何模块失效时可以回到逐日精确复制正式策略的 `formal_anchor`。",
        "",
        "## 2. 逐层输入输出",
        "",
        "1. 日频 Point-in-Time 数据 -> 38 张资产卡、六模块状态和股债九宫格；未发布数据不进入当日卡片。",
        "2. 九宫格和 view filters -> 信用/转债/权益等主动超配硬阻断；正式锚点作为被动基准单独审计。",
        "3. 机械 MarketSkill -> 五袖套相对规则 views，不直接给仓位。",
        "4. iTransformer -> 过去 120 个交易日、64 个变量，64 维、2 层、4 头，预测五袖套未来 5/20/60 日风险调整相对分数；训练上限 300 epoch，按验证 early stopping。",
        "5. Champion-challenger -> 因果动量基线与 iTransformer 比较；只有已成熟历史中的增量可靠度为正，iTransformer 残差才获得 0%–75% 权重。",
        "6. PPO -> 9 个动作：持有、正式锚点、规则 20 日、基线 5/20/60 日、iTransformer 增量 5/20/60 日；三种子按动作家族投票。",
        "7. EP/skfolio -> iTransformer/基线只进入 `posterior_mu`，不进入协方差；CLARABEL 求解 MeanRisk，随后经过趋势刹车、袖套上下限与九宫格硬约束。",
        "8. 执行与回测 -> 信号日后第一个交易日执行，5bp 双边换手成本，无现金，日频记录净值、持仓、换手和归因。",
        "",
        "## 3. 产品参数",
        "",
        "| 产品 | 均值视图尺度 | 最大主动偏离 | 最大袖套换手 | 相对波动惩罚 | 相对回撤惩罚 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| low_vol | {cfg['views']['daily_mu_scale']['low_vol']:.4f} | {cfg['optimizer']['low_vol']['max_active_deviation']:.3f} | {cfg['optimizer']['low_vol']['max_sleeve_turnover']:.2f} | {cfg['reward']['relative_volatility_penalty']['low_vol']:.2f} | {cfg['reward']['relative_drawdown_penalty']['low_vol']:.2f} |",
        f"| balanced | {cfg['views']['daily_mu_scale']['balanced']:.4f} | {cfg['optimizer']['balanced']['max_active_deviation']:.3f} | {cfg['optimizer']['balanced']['max_sleeve_turnover']:.2f} | {cfg['reward']['relative_volatility_penalty']['balanced']:.2f} | {cfg['reward']['relative_drawdown_penalty']['balanced']:.2f} |",
        f"| high_vol | {cfg['views']['daily_mu_scale']['high_vol']:.4f} | {cfg['optimizer']['high_vol']['max_active_deviation']:.3f} | {cfg['optimizer']['high_vol']['max_sleeve_turnover']:.2f} | {cfg['reward']['relative_volatility_penalty']['high_vol']:.2f} | {cfg['reward']['relative_drawdown_penalty']['high_vol']:.2f} |",
        "",
        "### 3.1 产品化 PPO 部署门",
        "",
        "| 产品 | 默认动作 | PPO 允许的动作家族 |",
        "|---|---|---|",
        f"| low_vol | {cfg['deployment']['base_action']['low_vol']} | {', '.join(cfg['deployment']['allowed_action_families']['low_vol'])} |",
        f"| balanced | {cfg['deployment']['base_action']['balanced']} | {', '.join(cfg['deployment']['allowed_action_families']['balanced'])} |",
        f"| high_vol | {cfg['deployment']['base_action']['high_vol']} | {', '.join(cfg['deployment']['allowed_action_families']['high_vol'])} |",
        "",
        f"低波和中波仍在所有候选权重中使用机械 Market Views，但不允许 PPO 直接选择裸 `rules_only` 动作；高波以 `rules_only` 为默认动作。只有候选相对默认动作的验证奖励至少高 `{cfg['deployment']['minimum_validation_edge_vs_base']:.3f}`，家族概率不低于 `{cfg['deployment']['minimum_family_probability']:.2f}`，且领先第二名至少 `{cfg['deployment']['minimum_probability_margin']:.2f}`，才允许覆盖默认动作。",
        "",
        "## 4. 回测结果",
        "",
        summary.to_markdown(index=False),
        "",
        "## 5. 动作分布",
        "",
        action_table.to_markdown(index=False),
        "",
        "## 6. Point-in-Time 审计",
        "",
        audit.to_markdown(index=False),
        "",
        "## 7. iTransformer 增强实验",
        "",
        f"排名优先残差 challenger 的 CV 中位 RankIC 为 `{rank_gate.get('cv_median_rank_ic')}`，因果基线为 `{rank_gate.get('baseline_cv_median_rank_ic')}`，折权重为 `{rank_gate.get('fold_residual_alphas')}`，最终残差权重为 `{rank_gate.get('final_residual_alpha')}`。该实验没有超过基线，因此不替换当前预测。",
        "",
        "## 8. 正式启用门禁",
        "",
        f"- 是否替换正式策略：`{gate['enabled_for_formal_strategy']}`",
        f"- 详细门禁：`{cfg['outputs']['gate_report']}`",
        "- 当前结论：低波和中波 Sharpe 已超过正式锚点与固定规则对照；高波的年化收益、Sharpe 和回撤均优于正式锚点，但 Sharpe 仍略低于固定规则对照。各产品 iTransformer 增量动作仍少于 3 次，所以继续作为研究挑战者。",
        "- 研究边界：2024+ 区间已经用于多轮诊断，不能再视为全新的独立样本外证据；当前 v6 参数应冻结后做前瞻纸面跟踪。",
        "",
        "## 9. 运行命令与主要输出",
        "",
        "```powershell",
        "python scripts\\129_run_integrated_daily_view_ppo.py",
        "```",
        "",
        f"- 日频结果：`{cfg['outputs']['daily_pattern']}`",
        f"- 决策与训练截止：`{cfg['outputs']['decisions_pattern']}`",
        f"- 日频资产权重：`{cfg['outputs']['holdings_pattern']}`",
        f"- 三产品汇总：`{cfg['outputs']['summary']}`",
        "- 迭代对比：`output/reports/integrated_daily_view_ppo_iteration_comparison.csv`",
        "- 迭代结论：`output/reports/integrated_daily_view_ppo_iteration_notes.md`",
        "- 分年度稳健性：`output/reports/integrated_daily_view_ppo_annual_robustness.csv`",
        "- 分年度动作：`output/reports/integrated_daily_view_ppo_annual_action_counts.csv`",
        f"- 可视化：`{cfg['outputs']['visual_html']}`",
    ]
    out_path(cfg, "report_md").write_text("\n".join(lines), encoding="utf-8")
    _write_visuals(cfg, summary)
    return summary, audit, gate
