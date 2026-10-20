"""Isolated v45 targets and teacher-guided PPO execution research.

Decisions are close-t targets, valued/executed at the next observed session.
Nothing in this module refreshes data or modifies a previous release.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.adaptive_daily_ppo import ActorCritic
from src.hierarchical_asset_itransformer import load_config as inherited_config, _deep_merge_config
from src.hierarchical_asset_portfolio_v7 import AdaptiveFractionEnvironment, ControllerContext, _strict_monthly_anchor
from src.integrated_daily_view_ppo import _base_card_states, _to_sleeve
from src.portfolio_accounting import ACCOUNTING_VERSION, accounting_return_panel, advance_holdings, calendar_decision_targets
from src.v1_strategy import load_config as v1_config
from src.v45_solver import RISK_SLEEVES, SolverContractError, allocate_assets, bounded_projection, causal_anchor, confidence_view_tilt, solve, valid

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("low_vol", "balanced", "high_vol")
LAYERS = ("A0", "A1", "A2", "A3")
CONTROL_LAYERS = ("R0", "R1")
START, END = pd.Timestamp("2022-01-05"), pd.Timestamp("2026-07-15")
HORIZONS = {5: .15, 20: .50, 60: .35}
VERSION = "v45_portfolio_execution_1"
NUMERICAL_ZERO = 1e-10


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(frame):
    frame = frame.copy()
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True, default=str) if isinstance(value, (list, dict, np.ndarray)) else value)
    h = hashlib.sha256(json.dumps([str(c) for c in frame.columns]).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def config(path=None):
    cfg = inherited_config(ROOT / "config/hierarchical_asset_itransformer_v44_accounting_fixed.yaml")
    override_path = Path(path or ROOT / "config/v45_research.yaml")
    if override_path.exists():
        cfg = _deep_merge_config(cfg, inherited_config(override_path))
    cfg = copy.deepcopy(cfg)
    # The accounting source is invariant even when an orchestration override is used.
    cfg["outputs"]["dataset_npz"] = "data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz"
    cfg["outputs"]["metadata"] = "output/reports/hierarchical_asset_itransformer_v43_clean_calendar_metadata.json"
    cfg["v45"] = {
        "predictions": "data_processed/v45_predictions.parquet",
        "forecast_gate": "output/reports/v45_forecast_gate.json",
        "forecast_metadata": "output/reports/v45_metadata.json",
        "monthly_targets": "data_processed/v45_anchor_smoothing_validated_targets.parquet",
        "monthly_ledger": "output/reports/v45_anchor_smoothing_anchor_ledger.csv",
    }
    cfg["controller"]["minimum_hold_days"] = cfg["portfolio"].get("minimum_hold_days", {"low_vol": 3, "balanced": 2, "high_vol": 1})
    if "asset_active_limits" in cfg["portfolio"]:
        cfg["portfolio"]["max_asset_active_weight"] = cfg["portfolio"]["asset_active_limits"]
    for old in ["enabled_only_after_forecast_gate", "learning_rate", "entropy_weight", "validation_fraction",
                "selection_validation_fraction", "validation_purge_days", "teacher_guidance"]:
        cfg["controller"].pop(old, None)
    return cfg


def output(profile, layer, kind, root=ROOT):
    suffix = "csv" if kind == "daily" else "parquet"
    folder = "output/reports" if kind == "daily" else "data_processed"
    return Path(root) / folder / f"v45_{profile}_{layer}_{kind}.{suffix}"


def frozen_accounting(cfg, root=ROOT):
    """Use the original return mask, never the feature-completion mask."""
    panel, modules, quotes, gaps = accounting_return_panel(Path(root), cfg["outputs"]["dataset_npz"], cfg["outputs"]["metadata"], max_stale_sessions=5)
    if len(panel.columns) != 38 or not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("Frozen v43 accounting requires 38 assets and a unique sorted calendar")
    return panel, modules, quotes, gaps


def prediction_adapter(frame, gate=None, metadata=None):
    """Return only causally issued, explicitly gated learned increments.

    Required base schema is the common v45 long parquet. Optional aliases make
    timing evidence compatible with the forecast agent, never infer it from zero.
    All seeds are averaged within a component; components are averaged within
    each horizon before the fixed (not renormalized) 21/h blend.
    """
    required = {"as_of_date", "task", "entity", "module", "horizon_days", "model", "seed",
                "prediction", "baseline_prediction", "active_increment"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"v45 predictions missing columns: {missing}")
    work = frame.copy()
    aliases = {"training_end": "train_end_date", "training_end_date": "train_end_date",
               "gate_date": "gate_as_of_date", "oof": "is_oof", "gate_passed": "component_gate_passed"}
    for old, new in aliases.items():
        if new not in work and old in work:
            work[new] = work[old]
    timing = ["train_end_date", "gate_as_of_date", "is_oof", "component_gate_passed"]
    unavailable = [c for c in timing if c not in work]
    # A global final-model pass alone is never historical gate evidence.
    if unavailable:
        if pd.to_numeric(work.active_increment, errors="coerce").fillna(0).ne(0).any():
            raise ValueError(f"Active v45 increments lack timing evidence: {unavailable}")
        return pd.DataFrame(columns=["as_of_date", "task", "entity", "increment_21", "active_components"]), {
            "status": "shadow_missing_timing_evidence", "missing_items": unavailable, "active_rows": 0}
    for column in ["as_of_date"] + [c for c in timing[:2] if c in work]:
        work[column] = pd.to_datetime(work[column], errors="raise")
    numeric = ["prediction", "baseline_prediction", "active_increment"]
    finite_prediction = np.isfinite(work[numeric].to_numpy(dtype=float)).all(axis=1)
    if work.duplicated(["as_of_date", "task", "entity", "horizon_days", "model", "seed"]).any():
        raise ValueError("Duplicate v45 prediction component")
    if not work.task.isin(["asset", "sleeve"]).all() or not work.horizon_days.isin(HORIZONS).all():
        raise ValueError("Invalid prediction task/horizon")
    learned = work.model.isin(["F1", "F2", "F3"])
    chosen = work.seed.eq(-1)
    if "selected_candidate" in work:
        chosen &= work.model.eq(work.selected_candidate)
    elif "selected_candidate" in (gate or {}):
        chosen &= work.model.eq(gate["selected_candidate"])
    else:
        chosen &= False
    live = learned & chosen & work.active_increment.fillna(0).ne(0)
    if (live & ~finite_prediction).any():
        raise ValueError("Nonfinite active v45 prediction/baseline/increment")
    if not unavailable:
        timing_ok = (work.train_end_date.lt(work.as_of_date) & work.gate_as_of_date.lt(work.as_of_date)
                     & work.is_oof.eq(True) & work.component_gate_passed.eq(True))
        if (live & ~timing_ok).any():
            raise ValueError("Active v45 increment fails OOF/train/gate time contract")
        live &= timing_ok
    if not learned.all() and work.loc[~learned, "active_increment"].ne(0).any():
        raise ValueError("F0/unknown model cannot supply active_increment")
    active = work.loc[live].copy()
    key = ["as_of_date", "task", "entity", "horizon_days", "model"]
    averaged = active.groupby(key, as_index=False).active_increment.mean()
    averaged = averaged.groupby(key[:-1], as_index=False).active_increment.mean()
    averaged["increment_21"] = averaged.active_increment * averaged.horizon_days.map(HORIZONS) * 21 / averaged.horizon_days
    result = averaged.groupby(key[:3], as_index=False).agg(increment_21=("increment_21", "sum"), active_components=("horizon_days", "size"))
    return result, {"status": "causal_gated" if len(active) else "no_active_learned_forecasts",
                    "missing_items": [], "active_rows": len(active), "input_rows": len(work),
                    "unavailable_prediction_rows": int((~finite_prediction).sum())}


def load_predictions(cfg, root=ROOT):
    paths = {k: Path(root) / cfg["v45"][k] for k in ["predictions", "forecast_gate", "forecast_metadata"]}
    if not paths["forecast_gate"].exists() and (Path(root) / "output/reports/v45_forecast_report.json").exists():
        paths["forecast_gate"] = Path(root) / "output/reports/v45_forecast_report.json"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return pd.DataFrame(), {"status": "forecast_pending", "active_rows": 0, "missing_items": missing}
    gate = json.loads(paths["forecast_gate"].read_text(encoding="utf-8"))
    frame = pd.read_parquet(paths["predictions"])
    if "train_end_date" not in frame and gate.get("folds"):
        # Producer v1 stores exact historical split indices in the fold manifest.
        arrays = np.load(Path(root) / cfg["data"].get("dataset", "data_processed/v45_dataset.npz"))
        dates = pd.DatetimeIndex(pd.to_datetime(arrays["dates"]))
        frame["as_of_date"] = pd.to_datetime(frame.as_of_date)
        frame["train_end_date"], frame["gate_as_of_date"] = pd.NaT, pd.NaT
        frame["is_oof"], frame["component_gate_passed"] = False, False
        for fold in gate["folds"]:
            split = fold.get("split", {})
            if not split.get("valid", False) or not fold.get("representation_contract_pass", False):
                continue
            train, test = np.asarray(split["train"], int), np.asarray(split["test"], int)
            cutoff = dates[int(split["cutoff_index"])]
            selected = fold.get("selected_candidate", fold.get("selected"))
            rows = frame.as_of_date.isin(dates[test]) & frame.model_signature.eq(fold["signature"])
            frame.loc[rows, "train_end_date"] = dates[train].max()
            frame.loc[rows, "gate_as_of_date"] = cutoff
            frame.loc[rows, "is_oof"] = True
            for component in fold.get("gates", []):
                if not component.get("active", False):
                    continue
                passing = rows & frame.task.eq(component["task"]) & frame.horizon_days.eq(component["horizon_days"])
                if component["task"] == "asset":
                    module = component["module"]
                    module = RISK_SLEEVES[int(module)] if isinstance(module, (int, float)) else module
                    passing &= frame.module.eq(module)
                passing &= frame.model.eq(component.get("candidate", selected)) & frame.seed.eq(-1)
                frame.loc[passing, "component_gate_passed"] = True
    return prediction_adapter(frame, gate, json.loads(paths["forecast_metadata"].read_text(encoding="utf-8")))


def dynamic_bounds(raw, row, bounds, trend_cfg, profile, audit=None):
    """Every guardrail is an input to BOTH anchor feasibility and utility solve."""
    result = {s: tuple(bounds[s]) for s in RISK_SLEEVES}
    blocks = {"credit": "state_grid_block_credit", "convertible_bonds": "state_grid_block_cb", "equity": "state_grid_block_equity"}
    for sleeve in RISK_SLEEVES:
        lo, hi = result[sleeve]
        blocked = bool(row.get(blocks.get(sleeve, ""), False))
        blocked |= sleeve in ["equity", "convertible_bonds"] and bool(row.get("state_liquidity_shock", False))
        if blocked:
            hi = min(hi, float(raw[sleeve]))
        if trend_cfg.get("enabled", True) and sleeve in trend_cfg["affected_sleeves"]:
            trend, threshold = float(row[f"state_trend_{sleeve}"]), float(trend_cfg["weak_threshold"])
            if trend < threshold:
                severity = np.clip((threshold - trend) / 1.5, 0, 1)
                hi = min(hi, max(lo, raw[sleeve] - trend_cfg["max_underweight_from_base"][profile] * severity))
        if 0 < lo-hi <= 1e-10:
            if audit is not None:
                audit[sleeve] = float(lo-hi)
            hi = lo
        result[sleeve] = (lo, hi)
    return result


def product_box_anchor(raw, bounds):
    """Resolve product-box roundoff only; economic conflicts require review."""
    values = raw.reindex(RISK_SLEEVES).to_numpy(float)
    lo, hi = (np.array([bounds[s][j] for s in RISK_SLEEVES]) for j in [0,1])
    if np.all(values >= lo) and np.all(values <= hi):
        return raw.copy()
    projected = bounded_projection(values, lo, hi)
    distance = float(np.abs(projected-values).sum())
    if distance > 1e-10:
        raise SolverContractError("infeasible", {"reason": "raw_product_box_conflict_requires_policy",
            "raw_anchor": raw.to_dict(), "product_projection_required_l1": distance,
            "bounds": bounds})
    return pd.Series(projected, index=RISK_SLEEVES)


def allocate_target(prior, sleeve_weights, modules, increment, profile, cfg):
    sleeve_weights = sleeve_weights.copy()
    sleeve_weights.loc[sleeve_weights.abs() < NUMERICAL_ZERO] = 0.
    out = pd.Series(0., index=prior.index)
    for sleeve in RISK_SLEEVES:
        assets = [a for a in prior.index if modules[a] == sleeve]
        if not assets:
            if sleeve_weights[sleeve] > 1e-10:
                raise ValueError(f"No frozen eligible assets for {sleeve}")
            continue
        scores = increment.reindex(assets, fill_value=0.)
        # Keep the v43 four-tenor strategic split exactly, including in A2.
        if sleeve == "rates":
            scores = scores * 0
        try:
            allocated = allocate_assets(prior.loc[assets], scores, pd.Series(0., index=assets),
                float(sleeve_weights[sleeve]), float(cfg["portfolio"]["asset_tilt_strength"][profile]),
                .75, float(cfg["portfolio"]["max_asset_active_weight"][profile]))
        except SolverContractError as exc:
            raise SolverContractError(exc.status, {**exc.detail, "sleeve": sleeve,
                "sleeve_weight": float(sleeve_weights[sleeve]), "prior": prior.loc[assets].to_dict()}) from exc
        out.loc[assets] = allocated
        if abs(allocated.sum() - sleeve_weights[sleeve]) > 1e-8:
            raise ValueError("Asset sleeve total invalid")
    if abs(out.sum() - 1) > 1e-8 or not np.isfinite(out).all():
        raise ValueError("Asset allocation budget/nonfinite")
    return out


def rule_states(cfg, calendar, root=ROOT):
    cards = pd.read_parquet(Path(root) / cfg["portfolio"]["frozen_state_cards"])
    cards["date"] = pd.to_datetime(cards.date)
    state = _base_card_states(cards).set_index("date").reindex(calendar)
    if state.isna().any().any():
        raise ValueError("Frozen state cards missing calendar dates; no unlimited carry")
    views = pd.read_parquet(Path(root) / cfg["portfolio"]["frozen_rule_views"])
    views["date"] = pd.to_datetime(views.date)
    if "validator_status" in views:
        views = views.loc[views.validator_status.eq("valid")]
    mapping = v1_config()
    tilts = {}
    for date, group in views.groupby("date"):
        pairs = pd.DataFrame({"source": group.source_asset.map(lambda x: _to_sleeve(str(x), mapping)),
            "target": group.target_asset.map(lambda x: _to_sleeve(str(x), mapping)), "confidence": group.confidence})
        pairs = pairs.loc[pairs.source.isin(RISK_SLEEVES) & pairs.target.isin(RISK_SLEEVES)]
        tilts[date] = confidence_view_tilt(pairs, 1.)
        legacy = pd.Series(0., index=RISK_SLEEVES)
        for pair in pairs.itertuples():
            if pair.source != pair.target:
                legacy[pair.source] += pair.confidence
                legacy[pair.target] -= pair.confidence
        legacy -= legacy.mean()
        norm = legacy.abs().max()
        legacy = legacy / norm if norm > 1e-12 else legacy*0
        for sleeve in RISK_SLEEVES:
            state.loc[date, f"state_rule_legacy_{sleeve}"] = legacy[sleeve]
    rule = pd.DataFrame.from_dict(tilts, orient="index").reindex(index=calendar, columns=RISK_SLEEVES).fillna(0.)
    for sleeve in RISK_SLEEVES:
        state[f"state_rule_view_{sleeve}"] = rule[sleeve]
        state[f"state_rule_legacy_{sleeve}"] = state[f"state_rule_legacy_{sleeve}"].fillna(0.)
    return state


def sleeve_return_history(returns, prior, modules):
    result = {}
    for sleeve in RISK_SLEEVES:
        assets = [a for a in prior if modules[a] == sleeve]
        weights = prior[assets].div(prior[assets].sum(axis=1).replace(0, np.nan), axis=0)
        # Missing assets with zero exposure are irrelevant; positive exposures
        # with stale (>5) valuation remain missing, not fake zero returns.
        contributions = (returns[assets] * weights).where(weights.ne(0), 0.)
        result[sleeve] = contributions.sum(axis=1, min_count=len(assets))
    return pd.DataFrame(result)


def target_signature(cfg, profile, forecast_status, root=ROOT):
    paths = [Path(root) / "src/v45_solver.py", Path(root) / "src/portfolio_accounting.py",
        Path(root) / "src/hierarchical_asset_portfolio_v7.py", Path(root) / "src/integrated_daily_view_ppo.py",
        Path(root) / "config/v1_strategy_config.yaml", Path(root) / cfg["portfolio"]["integrated_base_config"],
        Path(root) / cfg["outputs"]["dataset_npz"], Path(root) / cfg["outputs"]["metadata"],
        Path(root) / cfg["portfolio"]["formal_anchor_pattern"].format(profile=profile),
        Path(root) / cfg["portfolio"]["risk_budget_pattern"].format(profile=profile),
        Path(root) / cfg["portfolio"]["frozen_state_cards"], Path(root) / cfg["portfolio"]["frozen_rule_views"]]
    paths += [Path(root) / p for p in cfg["v45"].values() if (Path(root) / p).exists()]
    hashes = {str(p): sha_file(p) for p in paths}
    # PPO/reward edits cannot invalidate or influence this deterministic cache.
    own_target_code = "\n".join(inspect.getsource(f) for f in [frame_hash, frozen_accounting, prediction_adapter,
        load_predictions, dynamic_bounds, product_box_anchor, allocate_target, rule_states,
        sleeve_return_history, prepare_profile])
    hashes["v45_target_code"] = hashlib.sha256((own_target_code+str(HORIZONS)+str(NUMERICAL_ZERO)).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps({"hashes": hashes, "portfolio": cfg["portfolio"],
        "forecast": forecast_status, "paths": cfg["v45"]}, sort_keys=True).encode()).hexdigest()
    return signature, hashes


def target_cache_matches(profile, saved, signature, root=ROOT):
    if saved.get("signature") != signature:
        return False, "input/code signature changed"
    try:
        for layer in LAYERS + CONTROL_LAYERS:
            path = output(profile, layer, "targets", root)
            if not path.exists() or frame_hash(pd.read_parquet(path)) != saved["target_hashes"][layer]:
                return False, f"missing or changed target content: {path}"
    except Exception as exc:
        return False, f"unreadable target cache: {type(exc).__name__}: {exc}"
    return True, "validated cache"


def prepare_profile(profile, cfg, returns, quotes, predictions, forecast_status, resume=False, root=ROOT):
    root = Path(root)
    manifest_path = root / f"output/reports/v45_{profile}_target_manifest.json"
    signature, hashes = target_signature(cfg, profile, forecast_status, root)
    cache_invalidation = None
    if resume and manifest_path.exists():
        try:
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            matched, cache_invalidation = target_cache_matches(profile, saved, signature, root)
        except Exception as exc:
            matched, cache_invalidation = False, f"unreadable target manifest: {exc}"
        if matched:
            print(f"[v45/{profile}] reusing signature/content-validated target cache", flush=True)
            return saved
        print(f"[v45/{profile}] stale target cache: {cache_invalidation}; RECOMPUTING (no stale target reuse)", flush=True)
    calendar = returns.index
    prior, modules = _strict_monthly_anchor(profile, calendar, cfg)
    if not set(prior).issubset(returns.columns):
        raise ValueError("Frozen anchor asset absent from accounting source")
    monthly = pd.read_parquet(root / cfg["v45"]["monthly_targets"])
    monthly = monthly.loc[monthly.profile.eq(profile)].pivot(index="date", columns="sleeve", values="weight")
    monthly.index = pd.to_datetime(monthly.index)
    anchor = calendar_decision_targets(monthly[RISK_SLEEVES], calendar)
    risk = pd.read_parquet(root / cfg["portfolio"]["risk_budget_pattern"].format(profile=profile))
    risk["date"] = pd.to_datetime(risk.date)
    risk = calendar_decision_targets(risk.set_index("date")[RISK_SLEEVES], calendar)
    with (root / "config/v1_strategy_config.yaml").open(encoding="utf-8") as handle:
        product = yaml.safe_load(handle)["product"]["profiles"][profile]
    with (root / cfg["portfolio"]["integrated_base_config"]).open(encoding="utf-8") as handle:
        integrated = yaml.safe_load(handle)
    states = rule_states(cfg, calendar, root)
    sleeves = sleeve_return_history(returns, prior, modules)
    for sleeve in RISK_SLEEVES:
        parts = []
        for lookback in integrated["trend_brake"]["lookbacks_days"]:
            series = sleeves[sleeve].rolling(lookback, min_periods=max(20, lookback // 2))
            parts.append((series.sum() / (series.std(ddof=0) * np.sqrt(lookback) + 1e-8)).clip(-2, 2))
        states[f"state_trend_{sleeve}"] = pd.concat(parts, axis=1).mean(axis=1)
    dates = calendar[(calendar >= START - pd.Timedelta(days=10)) & (calendar <= END)]
    dates = dates[dates >= calendar[calendar.searchsorted(START) - 1]]
    if dates[1] != START or dates[-1] != END:
        raise ValueError("Required common portfolio calendar unavailable")
    bounds = {s: product["sleeve_bounds"][s] for s in RISK_SLEEVES}
    options = {**product["ep_skfolio_overrides"], **integrated["optimizer"][profile]}
    options["max_active_deviation"] = cfg["portfolio"].get("daily_active_limits", {}).get(profile, options["max_active_deviation"])
    options["max_sleeve_turnover"] = cfg["portfolio"].get("daily_target_change_limits", {}).get(profile, options["max_sleeve_turnover"])
    previous = {layer: None for layer in ["A0", "A1", "A2", "R0"]}
    targets = {layer: [] for layer in previous}
    diagnostics = {layer: [] for layer in previous}
    forecast = {} if predictions.empty else {d: g for d, g in predictions.groupby("as_of_date")}
    upstream = pd.read_csv(root / cfg["v45"]["monthly_ledger"])
    upstream = upstream.loc[upstream.profile.eq(profile)].copy()
    upstream["date"] = pd.to_datetime(upstream.date)
    ledger = calendar_decision_targets(upstream.pivot(index="date", columns="sleeve", values="raw_anchor")[RISK_SLEEVES], calendar)
    failure_context = {}
    try:
        for count, date in enumerate(dates):
            if count % 100 == 0:
                print(f"[v45/{profile}] target {count}/{len(dates)} {date.date()}", flush=True)
            history = sleeves.loc[:date].tail(252).dropna()
            if len(history) < 60 or not np.isfinite(history).all().all():
                raise ValueError(f"Insufficient valid covariance history {date}: {len(history)}")
            sample = history.cov().to_numpy()
            sigma = .8 * sample + .2 * np.diag(np.diag(sample))
            eigen_before = float(np.linalg.eigvalsh(sigma).min())
            jitter = max(1e-9, 1e-9 - eigen_before)
            covariance = pd.DataFrame((sigma + np.eye(5) * jitter) * 21, index=RISK_SLEEVES, columns=RISK_SLEEVES)
            row = states.loc[date]
            if row.filter(like="state_trend_").isna().any():
                raise ValueError(f"Unavailable trend state {date}")
            live = forecast.get(date, pd.DataFrame())
            asset_inc, sleeve_inc = pd.Series(dtype=float), pd.Series(0., index=RISK_SLEEVES)
            if not live.empty:
                asset_inc = live.loc[live.task.eq("asset")].set_index("entity").increment_21
                sleeve_values = live.loc[live.task.eq("sleeve")].set_index("entity").increment_21
                unknown = set(sleeve_values.index) - set(RISK_SLEEVES)
                if unknown:
                    raise ValueError(f"Unknown prediction sleeves: {unknown}")
                sleeve_inc.update(sleeve_values)
            investable_active = asset_inc.reindex([a for a in prior.columns if modules[a] != "rates"], fill_value=0.)
            active_count = int(sleeve_inc.ne(0).sum() + investable_active.ne(0).sum())
            for layer in previous:
                raw = (risk if layer == "A0" else anchor).loc[date].copy()
                raw[raw.abs() < 1e-12] = 0.
                if not np.isfinite(raw).all() or abs(raw.sum() - 1) > 1e-8:
                    raise ValueError(f"Invalid monthly sleeve anchor {profile}/{date}")
                p = raw if previous[layer] is None else previous[layer]
                product_anchor = product_box_anchor(raw, bounds)
                bound_snap = {}
                box = dynamic_bounds(product_anchor, row, bounds, integrated["trend_brake"], profile, bound_snap)
                failure_context = {"raw_anchor": raw.to_dict(), "product_feasible_anchor": product_anchor.to_dict(),
                    "previous_target": p.to_dict(), "bounds": box, "options": options}
                effective = causal_anchor(product_anchor, p, box, options["max_active_deviation"], options["max_sleeve_turnover"])
                mu = pd.Series(0., index=RISK_SLEEVES)
                if layer != "A0":
                    prefix = "state_rule_legacy" if layer == "R0" else "state_rule_view"
                    mu = row[[f"{prefix}_{s}" for s in RISK_SLEEVES]].set_axis(RISK_SLEEVES) * integrated["views"]["daily_mu_scale"][profile]
                if layer == "A2":
                    mu = mu + sleeve_inc
                result = solve(effective.weights, mu, covariance, p, box, history, options)
                failure_context["solver_raw_weights"] = result.weights.to_dict()
                cleaned = result.weights.copy()
                cleaned.loc[cleaned.abs() < NUMERICAL_ZERO] = 0.
                if not valid(cleaned.to_numpy(), effective.weights.to_numpy(), p.to_numpy(),
                    np.array([box[s][0] for s in RISK_SLEEVES]), np.array([box[s][1] for s in RISK_SLEEVES]),
                    options["max_active_deviation"], options["max_sleeve_turnover"]):
                    raise ValueError("Numerical zero cleanup failed original final sleeve contract")
                allocation = allocate_target(prior.loc[date], result.weights, modules,
                    asset_inc if layer == "A2" else pd.Series(dtype=float), profile, cfg)
                previous[layer] = result.weights.copy()
                targets[layer].append(allocation.rename(date))
                diagnostics[layer].append({"date": date, "profile": profile, "layer": layer,
                    **result.diagnostics, **effective.diagnostics,
                    "product_box_anchor_adjustment_l1": float((product_anchor-raw).abs().sum()),
                    "numerical_product_box_snap_l1": float((product_anchor-raw).abs().sum()),
                    "numerical_bounds_snap": json.dumps(bound_snap, sort_keys=True),
                    "original_raw_to_effective_l1": float((effective.weights-raw).abs().sum()),
                    "anchor_priority": "product_box_projection_then_guardrail_then_joint_causal_anchor",
                    "numerical_zero_cleanup_l1": float((cleaned-result.weights).abs().sum()),
                    "numerical_zero_cleanup_budget_residual": float(abs(cleaned.sum()-1)),
                    "asset_budget_residual": float(abs(allocation.sum()-1)),
                    "covariance_eigen_before": eigen_before, "covariance_jitter_daily": jitter,
                    "active_components": active_count if layer == "A2" else 0,
                    "forecast_available": date in forecast, "forecast_status": forecast_status["status"],
                    "forecast_confidence": float(active_count > 0) if layer == "A2" else 0.,
                    "effective_model_level": float(active_count > 0) if layer == "A2" else 0.,
                    "mean_prediction_uncertainty": 0.,
                    "target_asset_count": int(allocation.gt(1e-10).sum()), "investable_asset_count": len(prior.columns),
                    **{f"gated_score_{s}": float(sleeve_inc[s]) if layer == "A2" else 0. for s in RISK_SLEEVES},
                    **{f"gated_uncertainty_{s}": 0. for s in RISK_SLEEVES},
                    **{f"anchor_{s}": float(effective.weights[s]) for s in RISK_SLEEVES},
                    **{f"raw_anchor_{s}": float(raw[s]) for s in RISK_SLEEVES},
                    **{f"original_monthly_raw_{s}": float(ledger.loc[date, s]) for s in RISK_SLEEVES},
                    **{f"mu_21_{s}": float(mu[s]) for s in RISK_SLEEVES},
                    **{f"lower_{s}": box[s][0] for s in RISK_SLEEVES},
                    **{f"upper_{s}": box[s][1] for s in RISK_SLEEVES}})
    except Exception as exc:
        detail = getattr(exc, "detail", {"reason": str(exc)})
        write_json(root / f"output/reports/v45_{profile}_target_blocked.json",
                   {"profile": profile, "date": str(date), "layer": layer, **detail, **failure_context})
        raise
    target_hashes = {}
    for layer in LAYERS + CONTROL_LAYERS:
        source = {"A3": "A2", "R1": "A1"}.get(layer, layer)
        frame = pd.DataFrame(targets[source]).rename_axis("date").sort_index()
        path = output(profile, layer, "targets", root)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        target_hashes[layer] = frame_hash(frame)
        diag = pd.DataFrame(diagnostics[source]).set_index("date")
        diag.to_parquet(output(profile, layer, "diagnostics", root))
    states.loc[dates].to_parquet(root / f"data_processed/v45_{profile}_states.parquet")
    manifest = {"version": VERSION, "signature": signature, "inputs": hashes,
        "cache_invalidation": cache_invalidation,
        "profile": profile, "forecast": forecast_status, "target_hashes": target_hashes,
        "a2_a3_same_target": target_hashes["A2"] == target_hashes["A3"],
        "input_asset_count": 38, "investable_assets": list(prior.columns), "module_map": modules,
        "decision_start": str(dates[0]), "execution_start": str(START), "execution_end": str(END),
        "calendar_hash": frame_hash(pd.DataFrame(index=dates)),
        "state_columns": [c for c in states if c.startswith("state_")],
        "active_days": int(pd.DataFrame(diagnostics["A2"]).active_components.gt(0).sum()),
        "R0_R1_scope": "same frozen monthly anchors; R0 legacy daily confidence normalization, R1=A1 confidence-preserving daily views; no claim of monthly confidence counterfactual",
        "A0_scope": "frozen raw monthly risk-budget anchor, zero-mu constrained covariance utility with common daily guardrails and causal anchor policy; not unmodified risk parity",
        "scope": "close_to_close_research_not_open_price_execution"}
    write_json(manifest_path, manifest)
    blocked_path = root / f"output/reports/v45_{profile}_target_blocked.json"
    if blocked_path.exists():
        prior_failure = json.loads(blocked_path.read_text(encoding="utf-8"))
        write_json(blocked_path, {**prior_failure, "original_status": prior_failure.get("original_status", prior_failure.get("status")),
            "status": "resolved", "resolved_target_signature": signature})
    return manifest


@dataclass
class ExecutionContext(ControllerContext):
    mechanical_age: int = 0
    mechanical_last_action: int = 0


class ExecutionEnvironment(AdaptiveFractionEnvironment):
    """Same observations/mechanical policy, but a dynamic same-target benchmark.

    No mutable rollout state is stored on the environment. Context contains both
    books, so teacher branches, validation and sampled PPO trajectories cannot
    leak holdings, NAVs or trade ages into one another.
    """
    def __init__(self, profile, cfg, target, diagnostics, state, anchor, returns, quotes, modules, initial=None):
        self.profile, self.cfg = profile, cfg
        self.target = target.copy()
        self.assets = list(target.columns)
        self.module_map = modules
        self.dates = pd.DatetimeIndex(target.index)
        self.returns = returns.reindex(index=self.dates, columns=self.assets)
        self.tradable = quotes.reindex(index=self.dates, columns=self.assets)
        if self.tradable.isna().any().any():
            raise ValueError("Missing quote eligibility")
        self.diagnostics, self.state = diagnostics.reindex(self.dates), state.reindex(self.dates).copy()
        self.formal = anchor.reindex(index=self.dates, columns=self.assets)
        self.initial = target.iloc[0].to_numpy().copy() if initial is None else np.asarray(initial).copy()
        self.actions = np.asarray(cfg["controller"]["action_fractions"], dtype=float)
        if not np.array_equal(self.actions, [0., .25, .5, 1.]):
            raise ValueError("Invalid v45 execution action grid")
        self.cost_rate = cfg["portfolio"]["transaction_cost_bps"] / 10000
        self.state["state_quote_available_fraction"] = self.tradable.mean(axis=1)
        self.state["state_forecast_live"] = self.diagnostics.active_components.gt(0).astype(float)
        self.state_cols = [c for c in self.state if c.startswith("state_")]
        if not np.isfinite(self.state[self.state_cols].to_numpy(float)).all():
            raise ValueError("Nonfinite/missing observed state")
        for name, frame in [("target", self.target), ("anchor", self.formal)]:
            if not np.isfinite(frame.to_numpy()).all() or (frame < -1e-10).any().any() or not np.allclose(frame.sum(axis=1), 1, atol=1e-8, rtol=0):
                raise ValueError(f"Invalid execution {name}")
        self._target = self.target.to_numpy(float)
        self._returns = self.returns.to_numpy(float)
        self._quotes = self.tradable.to_numpy(bool)
        self._anchor = self.formal.to_numpy(float)
        self._state = self.state[self.state_cols].to_numpy(float)
        self._module_positions = [[i for i, a in enumerate(self.assets) if modules[a] == s] for s in RISK_SLEEVES]

    @staticmethod
    def initial_context():
        return ExecutionContext()

    def initial_weights(self, index):
        return self.initial.copy() if index == 0 else self._anchor[index].copy()

    def _sleeves(self, weights):
        return np.array([weights[p].sum() for p in self._module_positions])

    def observation(self, index, current, context, mean, std):
        static = np.clip((self._state[index] - mean) / std, -6, 6)
        diag = self.diagnostics.iloc[index]
        current_sleeve, target_sleeve = self._sleeves(current), self._sleeves(self._target[index])
        anchor_sleeve = self._sleeves(self._anchor[index])
        dynamic = np.r_[current_sleeve, target_sleeve, anchor_sleeve,
            [diag.get(f"gated_score_{s}", 0.) for s in RISK_SLEEVES],
            [diag.get(f"gated_uncertainty_{s}", 0.) for s in RISK_SLEEVES],
            np.abs(self._target[index] - current).sum(), diag.get("forecast_confidence", 0.),
            math.log1p(context.age) / math.log(253), context.nav / context.peak - 1,
            math.sqrt(max(context.ewma_variance, 0.) * 252),
            context.since_trade_nav / context.since_trade_formal_nav - 1,
            [float(context.last_action == a) for a in range(4)]]
        obs = np.r_[static, dynamic].astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("Nonfinite execution observation")
        return obs

    def allowed_action(self, index, action, context):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.actions):
            raise ValueError("Invalid PPO action index")
        minimum = self.cfg["controller"]["minimum_hold_days"][self.profile]
        if context.age < minimum and not bool(self.state.iloc[index].get("state_liquidity_shock", False)):
            return 0
        return int(action)

    def execution_residuals(self, index, weights, detailed=False):
        diag = self.diagnostics.iloc[index]
        sleeve = self._sleeves(weights)
        bounds_residual, concentration, asset_active = 0., 0., 0.
        active_limit = self.cfg["portfolio"]["max_asset_active_weight"][self.profile]
        components = {}
        for j, name in enumerate(RISK_SLEEVES):
            violation = max(0., float(diag.get(f"lower_{name}", 0.))-sleeve[j], sleeve[j]-float(diag.get(f"upper_{name}", 1.)))
            bounds_residual = max(bounds_residual, violation)
            components[f"sleeve_bounds:{name}"] = float(violation)
            positions = self._module_positions[j]
            if not positions:
                continue
            w, anchor = weights[positions], self._anchor[index, positions]
            concentration = max(concentration, float(np.max(w-.75*sleeve[j])))
            for pos, weight in zip(positions, w):
                components[f"concentration:{self.assets[pos]}"] = max(0., float(weight-.75*sleeve[j]))
            if anchor.sum() > 1e-12:
                capital_anchor = anchor/anchor.sum()*sleeve[j]
                asset_active = max(asset_active, float(np.max(np.abs(w-capital_anchor)-active_limit)))
                for pos, weight, base in zip(positions, w, capital_anchor):
                    components[f"asset_active:{self.assets[pos]}"] = max(0., float(abs(weight-base)-active_limit))
        result = {"budget": float(abs(weights.sum()-1)), "short": max(0., float(-weights.min())),
            "sleeve_bounds": max(0., bounds_residual), "concentration": max(0., concentration),
            "asset_active": max(0., asset_active)}
        return {**result, **components} if detailed else result

    def execute(self, index, current, action):
        """Block a quote mapping that creates/worsens a hard-book violation.

        Fully invested quote freezing rescales the tradable side, which is not
        necessarily a convex move toward the target. A whole-order no-fill is
        conservative, preserves frozen positions, and introduces no risk override.
        Existing drift violations are disclosed even on an unchanged book.
        """
        before = self.execution_residuals(index, current, detailed=True)
        proposal = advance_holdings(current, self._target[index], self.actions[action], self._returns[index+1], self.cost_rate, self._quotes[index+1])
        proposed = self.execution_residuals(index, proposal["weights"], detailed=True)
        worsened = {k: {"before": before[k], "proposed": proposed[k]} for k in proposed
                    if ":" in k and proposed[k] > max(before[k], 1e-8)+1e-10}
        blocked = bool(proposal["unfilled"] and worsened)
        if blocked:
            actual = advance_holdings(current, current, 0., self._returns[index+1], self.cost_rate, self._quotes[index+1])
            actual["unfilled"] = True
        else:
            actual = proposal
        executed = self.execution_residuals(index, actual["weights"], detailed=True)
        end = self.execution_residuals(index, actual["end_weights"], detailed=True)
        audit = {f"{prefix}_{key}_residual": value for prefix, residual in
                 [("pretrade", before), ("proposed", proposed), ("executed", executed), ("end", end)] for key, value in residual.items() if ":" not in key}
        audit.update(quote_order_blocked=bool(blocked),
            quote_violation_detail=json.dumps(worsened if blocked else {}, sort_keys=True),
            executed_violation_detail=json.dumps({k:v for k,v in executed.items() if ":" in k and v>1e-8}, sort_keys=True),
            end_violation_detail=json.dumps({k:v for k,v in end.items() if ":" in k and v>1e-8}, sort_keys=True),
            execution_constraint_reason="quote_mapping_would_worsen_hard_book_violation" if blocked else
                ("preexisting_drift_or_incomplete_risk_correction" if any(v>1e-8 for v in executed.values()) else "validated"),
            unfinished_risk_correction=any(v>1e-8 for v in end.values()))
        return actual, audit, 0 if blocked else action

    def step(self, index, action, current, context):
        if index + 1 >= len(self.dates):
            raise IndexError("No next execution session")
        action = self.allowed_action(index, action, context)
        # Benchmark mechanical decision uses its own age and own drifted book.
        mechanical_current = self.initial_weights(index) if context.formal_weights is None else context.formal_weights.copy()
        mechanical_context = copy.copy(context)
        mechanical_context.age, mechanical_context.last_action = context.mechanical_age, context.mechanical_last_action
        mechanical_action = self.mechanical_action(index, mechanical_current, mechanical_context)
        requested_action = action
        candidate, execution_audit, action = self.execute(index, current, action)
        benchmark, benchmark_audit, mechanical_action = self.execute(index, mechanical_current, mechanical_action)
        r, m = candidate["return"], benchmark["return"]
        beta = .94
        variance = beta * context.ewma_variance + (1 - beta) * r*r
        mechanical_variance = beta * context.formal_ewma_variance + (1 - beta) * m*m
        nav, mechanical_nav = context.nav * (1+r), context.formal_nav * (1+m)
        peak, mechanical_peak = max(nav, context.peak), max(mechanical_nav, context.formal_peak)
        downside = .35 * max(0., max(0., -r) - max(0., -m))
        vol_penalty = {"low_vol": .75, "balanced": .5, "high_vol": .2}[self.profile] * max(0., math.sqrt(variance*252) - math.sqrt(mechanical_variance*252)) * (1-beta)
        dd_penalty = {"low_vol": .5, "balanced": .35, "high_vol": .15}[self.profile] * max(0., abs(nav/peak-1) - abs(mechanical_nav/mechanical_peak-1))
        active = float(np.abs(candidate["weights"] - self._anchor[index]).sum())
        active_penalty = self.cfg["controller"]["active_weight_penalty"] * active
        reward = 100 * (r - m - downside - vol_penalty - dd_penalty - active_penalty)
        traded, mechanical_traded = candidate["turnover"] > 1e-8, benchmark["turnover"] > 1e-8
        new = ExecutionContext(age=0 if traded else context.age+1,
            last_action=action if traded else context.last_action,
            mechanical_age=0 if mechanical_traded else context.mechanical_age+1,
            mechanical_last_action=mechanical_action if mechanical_traded else context.mechanical_last_action,
            nav=nav, formal_nav=mechanical_nav, peak=peak, formal_peak=mechanical_peak,
            since_trade_nav=(1+r) if traded else context.since_trade_nav*(1+r),
            since_trade_formal_nav=(1+m) if traded else context.since_trade_formal_nav*(1+m),
            ewma_variance=variance, formal_ewma_variance=mechanical_variance,
            formal_weights=benchmark["end_weights"].copy())
        info = {"date": self.dates[index+1], "signal_date": self.dates[index],
            "strategy_return": r, "gross_return": candidate["gross_return"], "nav": nav,
            "mechanical_return": m, "mechanical_nav": mechanical_nav,
            "turnover": candidate["turnover"], "transaction_cost": candidate["cost"],
            "cost": candidate["cost"], "mechanical_turnover": benchmark["turnover"],
            "unfilled": candidate["unfilled"], "quote_frozen_count": int((~self._quotes[index+1]).sum()),
            "action": action, "action_fraction": float(self.actions[action]),
            "requested_action": requested_action, "requested_action_fraction": float(self.actions[requested_action]),
            **execution_audit, "mechanical_quote_order_blocked": benchmark_audit["quote_order_blocked"],
            "target_asset_count": int((self._target[index]>1e-10).sum()),
            "held_asset_count": int((candidate["weights"]>1e-10).sum()),
            "reward": reward, "reward_return_edge": 100*(r-m),
            "reward_downside_penalty": 100*downside, "reward_volatility_penalty": 100*vol_penalty,
            "reward_drawdown_penalty": 100*dd_penalty, "reward_active_penalty": 100*active_penalty,
            "active_l1": active, "pretrade_weights": current.copy(),
            "weights": candidate["weights"].copy(), "end_weights": candidate["end_weights"].copy()}
        if not np.isfinite([reward, nav, mechanical_nav]).all():
            raise ValueError("Nonfinite reward/NAV")
        return reward, index+1, candidate["end_weights"].copy(), new, info

    def teacher_action(self, index, current, horizon_days=5, context=None):
        if index + horizon_days >= len(self.dates):
            raise IndexError("Unmatured teacher label")
        utilities = []
        for action in range(4):
            held = current.copy()
            branch = copy.deepcopy(context if context is not None else self.initial_context())
            utility = 0.
            for offset in range(horizon_days):
                reward, _, held, branch, _ = self.step(index+offset, action if offset == 0 else 0, held, branch)
                utility += reward
            utilities.append(utility)
        return int(np.argmax(utilities))


def load_environment(profile, layer, cfg, returns, quotes, root=ROOT):
    root = Path(root)
    manifest = json.loads((root / f"output/reports/v45_{profile}_target_manifest.json").read_text(encoding="utf-8"))
    target = pd.read_parquet(output(profile, layer, "targets", root))
    if frame_hash(target) != manifest["target_hashes"][layer]:
        raise ValueError("Target content hash mismatch")
    diagnostics = pd.read_parquet(output(profile, layer, "diagnostics", root))
    states = pd.read_parquet(root / f"data_processed/v45_{profile}_states.parquet")
    prior, modules = _strict_monthly_anchor(profile, returns.index, cfg)
    anchor = prior.reindex(target.index).copy()
    for s in RISK_SLEEVES:
        assets = [a for a in anchor if modules[a] == s]
        shares = anchor[assets].div(anchor[assets].sum(axis=1).replace(0, np.nan), axis=0)
        anchor.loc[:, assets] = shares.mul(diagnostics[f"anchor_{s}"], axis=0)
    initial = pd.read_parquet(output(profile, "A0", "targets", root)).iloc[0].reindex(target.columns).to_numpy()
    return ExecutionEnvironment(profile, cfg, target, diagnostics, states, anchor, returns, quotes, modules, initial)


def split_history(env, year):
    eligible = env.eligible(pd.Timestamp(year, 1, 1))
    ctl = env.cfg["controller"]
    ns = max(30, math.ceil(len(eligible) * ctl.get("selection_fraction", .1)))
    ng = max(30, math.ceil(len(eligible) * ctl.get("deployment_gate_fraction", .1)))
    purge = max(20, ctl.get("purge_days", 20), ctl.get("teacher_horizon_days", 5)+1)
    train_end = len(eligible) - ns - ng - 2*purge
    if train_end < ctl.get("minimum_train_days", 252):
        return None
    train = eligible[:train_end]
    selection = eligible[train_end+purge:train_end+purge+ns]
    gate = eligible[train_end+purge+ns+purge:]
    if len(gate) < ng:
        raise ValueError("Incorrect nested PPO split")
    return train, selection, gate


def normalization(env, train):
    static = env._state[train]
    mean, std = static.mean(axis=0), static.std(axis=0)
    return mean, np.where(std > 1e-8, std, 1.)


def simulate(env, rows, model=None, mean=None, std=None):
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0 or np.any(np.diff(rows) != 1):
        raise ValueError("Rollout requires a nonempty contiguous calendar")
    current, context = env.initial_weights(int(rows[0])), env.initial_context()
    result = []
    for i in rows:
        if model is None:
            action = env.mechanical_action(i, current, context)
        else:
            with torch.no_grad():
                logits, _ = model(torch.tensor(env.observation(i, current, context, mean, std))[None])
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite policy logits")
            action = int(logits.argmax(-1).item())
        _, _, current, context, info = env.step(i, action, current, context)
        result.append(info)
    return result


def gate_model(env, rows, model, mean, std):
    edges, net_edges = [], []
    ctl = env.cfg["controller"]
    for block in np.array_split(rows, ctl.get("deployment_gate_blocks", 3)):
        if len(block) < 5:
            continue
        policy, mechanical = simulate(env, block, model, mean, std), simulate(env, block)
        edges.append(float(np.mean([p["reward"] - m["reward"] for p, m in zip(policy, mechanical)])))
        net_edges.append(float(np.mean([p["strategy_return"] - m["strategy_return"] for p, m in zip(policy, mechanical)])))
    passing = (len(edges) >= ctl.get("minimum_deployment_gate_blocks", 3)
        and np.mean(np.array(edges) > 0) >= ctl.get("minimum_deployment_positive_fraction", 2/3)-1e-7
        and np.mean(edges) >= ctl["minimum_validation_edge"]
        and min(edges) >= ctl.get("minimum_deployment_worst_edge", -.002))
    return {"passed": bool(passing), "block_edges_reward_units": edges,
        "block_edges_net_return": net_edges, "blocks": len(edges)}


def model_signature(env, year, lr, entropy):
    h = hashlib.sha256()
    for file in [Path(__file__), ROOT / "src/v45_solver.py", ROOT / "src/portfolio_accounting.py", ROOT / "src/adaptive_daily_ppo.py"]:
        h.update(file.read_bytes())
    h.update(json.dumps({"version": VERSION, "accounting": ACCOUNTING_VERSION,
        "cfg": env.cfg, "year": year, "lr": lr, "entropy": entropy}, sort_keys=True).encode())
    for frame in [env.target, env.formal, env.state[env.state_cols], env.diagnostics, env.returns, env.tradable]:
        h.update(frame_hash(frame.loc[frame.index < pd.Timestamp(year, 1, 1)]).encode())
    return h.hexdigest()


def finite_update(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite PPO loss; seed stopped")
    optimizer.zero_grad()
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Nonfinite PPO gradient; seed stopped")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite PPO parameters; seed stopped")


def fit_seed(env, year, seed, lr, entropy_weight, split, log_callback=None):
    """Actual categorical PPO + GAE, with matured five-session teacher labels."""
    ctl = env.cfg["controller"]
    train, selection, _ = split
    torch.manual_seed(seed)
    mean, std = normalization(env, train)
    probe = env.observation(int(train[0]), env.initial_weights(int(train[0])), env.initial_context(), mean, std)
    model = ActorCritic(len(probe), 4, ctl["hidden_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    teacher_obs, teacher_actions, teacher_audit = [], [], []
    horizon = int(ctl.get("teacher_horizon_days", 5))
    current, context = env.initial_weights(int(train[0])), env.initial_context()
    train_set = set(train)
    for i in train:
        if all(i+j in train_set for j in range(horizon)) and i+horizon <= train[-1]+1:
            teacher_obs.append(env.observation(i, current, context, mean, std))
            teacher_actions.append(env.teacher_action(i, current, horizon, context))
            teacher_audit.append({"signal_date": env.dates[i], "label_end_date": env.dates[i+horizon], "train_last_execution": env.dates[train[-1]+1]})
        action = env.mechanical_action(i, current, context)
        _, _, current, context, _ = env.step(i, action, current, context)
    if not teacher_obs:
        raise ValueError("No fully matured PPO teacher labels")
    teacher_x = torch.tensor(np.vstack(teacher_obs), dtype=torch.float32)
    teacher_y = torch.tensor(teacher_actions, dtype=torch.long)
    for _ in range(ctl.get("teacher_pretrain_epochs", 30)):
        logits, _ = model(teacher_x)
        finite_update(F.cross_entropy(logits, teacher_y), model, optimizer)
    baseline = float(np.mean([r["reward"] for r in simulate(env, selection)]))
    best, best_state, patience = -np.inf, copy.deepcopy(model.state_dict()), 0
    logs = []
    started = time.monotonic()
    for epoch in range(1, min(120, ctl["max_epochs"])+1):
        current, context = env.initial_weights(int(train[0])), env.initial_context()
        obs_list, actions, rewards, values, old_logp = [], [], [], [], []
        for i in train:
            obs = env.observation(i, current, context, mean, std)
            with torch.no_grad():
                logits, value = model(torch.tensor(obs)[None])
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                logp = distribution.log_prob(action)
            reward, _, current, context, _ = env.step(i, int(action.item()), current, context)
            obs_list.append(obs)
            actions.append(int(action.item()))
            rewards.append(reward)
            values.append(float(value.item()))
            old_logp.append(float(logp.item()))
        advantage = np.zeros(len(train))
        last, next_value = 0., 0.
        for j in reversed(range(len(train))):
            delta = rewards[j] + ctl["gamma"]*next_value - values[j]
            last = delta + ctl["gamma"]*ctl["gae_lambda"]*last
            advantage[j], next_value = last, values[j]
        target = torch.tensor(advantage + values, dtype=torch.float32)
        adv = torch.tensor((advantage-advantage.mean())/(advantage.std()+1e-8), dtype=torch.float32)
        x, a, old = torch.tensor(np.vstack(obs_list)), torch.tensor(actions), torch.tensor(old_logp)
        for _ in range(ctl["updates_per_epoch"]):
            logits, value = model(x)
            distribution = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(distribution.log_prob(a) - old)
            clipped = ratio.clamp(1-ctl["clip_ratio"], 1+ctl["clip_ratio"])
            policy_loss = -torch.minimum(ratio*adv, clipped*adv).mean()
            value_loss = (value-target).square().mean()
            teacher_logits, _ = model(teacher_x)
            teacher_loss = F.cross_entropy(teacher_logits, teacher_y)
            entropy = distribution.entropy().mean()
            loss = policy_loss + ctl["value_loss_weight"]*value_loss - entropy_weight*entropy + ctl.get("teacher_policy_loss_weight", .15)*teacher_loss
            finite_update(loss, model, optimizer)
        record = {"profile": env.profile, "year": year, "seed": seed, "lr": lr, "entropy_weight": entropy_weight,
            "epoch": epoch, "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()), "teacher_loss": float(teacher_loss.detach()),
            "entropy": float(entropy.detach()), "rollout_reward": float(sum(rewards)),
            "elapsed_seconds": time.monotonic()-started}
        if epoch == 1 or epoch % 5 == 0:
            val = float(np.mean([r["reward"] for r in simulate(env, selection, model, mean, std)]))
            record.update(selection_reward=val, mechanical_selection_reward=baseline)
            if val > best + .0001:
                best, best_state, patience = val, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
            print(f"[v45/{env.profile}/{year}/{seed} lr={lr} ent={entropy_weight}] epoch={epoch} selection={val:.6g} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        logs.append(record)
        if log_callback:
            log_callback(record)
        if patience >= ctl.get("patience", 8):
            break
    model.load_state_dict(best_state)
    return {"state_dict": best_state, "mean": mean, "std": std, "obs_dim": len(probe), "seed": seed,
        "selection_reward": best, "mechanical_selection_reward": baseline, "lr": lr, "entropy_weight": entropy_weight,
        "training_cutoff": str(pd.Timestamp(year, 1, 1)), "teacher_count": len(teacher_obs),
        "teacher_class_counts": np.bincount(teacher_actions, minlength=4).tolist(),
        "teacher_audit": teacher_audit, "signature": model_signature(env, year, lr, entropy_weight)}, logs


def restore_model(record, hidden):
    model = ActorCritic(record["obs_dim"], 4, hidden)
    model.load_state_dict(record["state_dict"])
    model.eval()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite model checkpoint")
    return model


def train_fold(env, year, resume=False, root=ROOT):
    split = split_history(env, year)
    if split is None:
        return [], {"year": year, "reason": "insufficient_training_history", "shadow": True}
    ctl = env.cfg["controller"]
    torch.set_num_threads(ctl["torch_threads"])
    folder = Path(root) / "data_processed/v45_controller_models" / env.profile / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    logs_path = Path(root) / f"output/reports/v45_{env.profile}_{year}_ppo_training.jsonl"
    def append_log(record):
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
    candidates = []
    for lr in ctl.get("learning_rates", [1e-4, 3e-4]):
        for entropy in ctl.get("entropy_weights", [.004, .01]):
            records = []
            for seed in ctl["seeds"]:
                path = folder / f"lr{lr}_entropy{entropy}_seed{seed}.pt"
                signature = model_signature(env, year, lr, entropy)
                if resume and path.exists():
                    record = torch.load(path, map_location="cpu", weights_only=False)
                    if record["signature"] != signature:
                        raise ValueError(f"Stale PPO checkpoint {path}; explicit retrain required")
                    restore_model(record, ctl["hidden_dim"])
                else:
                    record, _ = fit_seed(env, year, seed, lr, entropy, split, append_log)
                    torch.save(record, path)
                records.append(record)
            # Independent gate segment is never read for these four candidates.
            score = float(np.mean([r["selection_reward"]-r["mechanical_selection_reward"] for r in records]))
            candidates.append((score, lr, entropy, records))
    winner = max(candidates, key=lambda item: item[0])
    approved, gates = [], []
    for record in winner[3]:
        model = restore_model(record, ctl["hidden_dim"])
        gate = gate_model(env, split[2], model, record["mean"], record["std"])
        gates.append({"seed": record["seed"], **gate})
        if gate["passed"]:
            approved.append((record, model))
    fold = {"year": year, "selected_lr": winner[1], "selected_entropy": winner[2],
        "selection_edge": winner[0], "candidate_selection_edges": [{"lr": c[1], "entropy": c[2], "edge": c[0]} for c in candidates],
        "qualified_seeds": len(approved), "seed_gates": gates,
        "deployment_gate_passed": len(approved) >= ctl["minimum_qualified_models"],
        "split": {name: {"signal_start": str(env.dates[rows[0]]), "last_execution": str(env.dates[rows[-1]+1]), "rows": len(rows)}
                  for name, rows in zip(["train", "selection", "gate"], split)},
        "shadow": not bool(env.diagnostics.loc[env.diagnostics.index.year == year, "active_components"].gt(0).any()),
        "reason": "trained_shadow_requires_daily_active_forecast_and_execution_gate"}
    write_json(Path(root) / f"output/reports/v45_{env.profile}_{year}_ppo_gate.json", fold)
    return approved if fold["deployment_gate_passed"] else [], fold


def replay(env, models_by_year=None):
    models_by_year = models_by_year or {}
    current, context = env.initial_weights(0), env.initial_context()
    rows, holdings = [], []
    for index in range(len(env.dates)-1):
        models = models_by_year.get(env.dates[index].year, [])
        active = env.diagnostics.iloc[index].active_components > 0
        action = env.mechanical_action(index, current, context)
        reason, used = "mechanical_baseline", False
        if models_by_year:
            reason = "no_active_learned_forecast" if not active else "no_qualified_PPO_or_no_training_fallback"
            if active and len(models) >= env.cfg["controller"]["minimum_qualified_models"]:
                votes = []
                for record, model in models:
                    if not pd.Timestamp(record["training_cutoff"]) <= env.dates[index]:
                        raise ValueError("PPO model time contract violated")
                    with torch.no_grad():
                        logits, _ = model(torch.tensor(env.observation(index, current, context, record["mean"], record["std"]))[None])
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite deployed model")
                    votes.append(int(logits.argmax(-1).item()))
                counts = np.bincount(votes, minlength=4)
                if counts.max() >= 2:
                    action, reason, used = int(counts.argmax()), "qualified_ensemble", True
                else:
                    reason = "ensemble_disagreement"
        _, _, current, context, info = env.step(index, action, current, context)
        weights, end_weights = info.pop("weights"), info.pop("end_weights")
        pretrade = info.pop("pretrade_weights")
        info.update(model_used=used, shadow=not used, fallback_reason=reason,
            active_components=int(env.diagnostics.iloc[index].active_components),
            forecast_available=bool(env.diagnostics.iloc[index].get("forecast_available", False)))
        rows.append(info)
        for j, asset in enumerate(env.assets):
            holdings.append({"date": env.dates[index+1], "asset_code": asset, "module": env.module_map[asset],
                "signal_date": env.dates[index], "pretrade_weight": float(pretrade[j]),
                "target_weight": float(env._target[index, j]), "anchor_weight": float(env._anchor[index, j]),
                "quoted": bool(env._quotes[index+1,j]),
                "weight": float(weights[j]), "end_weight": float(end_weights[j]),
                "return_contribution": float(weights[j] * env._returns[index+1, j]) if np.isfinite(env._returns[index+1,j]) else 0.})
    daily = pd.DataFrame(rows)
    return daily, pd.DataFrame(holdings)


def statistics(values):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Invalid performance return sample")
    nav = np.cumprod(1+values)
    annual = float(np.expm1(np.log1p(values).mean()*252))
    vol = float(values.std(ddof=1)*np.sqrt(252)) if len(values) > 1 else 0.
    sharpe = float(values.mean()*252/vol) if vol > 1e-14 else 0.
    peak = np.maximum.accumulate(np.r_[1., nav])[1:]
    return {"annual_return": annual, "annual_volatility": vol, "sharpe": sharpe,
            "max_drawdown": float(np.min(nav/peak-1))}


def bootstrap_comparison(candidate, baseline, block_days=20, draws=3000, seed=20260727):
    p, b = np.asarray(candidate, float), np.asarray(baseline, float)
    if len(p) != len(b) or not len(p) or not np.isfinite(np.r_[p,b]).all():
        raise ValueError("Bootstrap requires finite same-day paired returns")
    rng = np.random.default_rng(seed)
    n = len(p)
    annual_delta, sharpe_delta = [], []
    for left in range(0, draws, 100):
        starts = rng.integers(0, n, size=(min(100, draws-left), math.ceil(n/block_days)))
        indices = ((starts[..., None]+np.arange(block_days)) % n).reshape(len(starts), -1)[:, :n]
        x, y = p[indices], b[indices]
        annual_delta.extend(np.expm1(np.log1p(x).mean(axis=1)*252)-np.expm1(np.log1p(y).mean(axis=1)*252))
        sx = np.divide(x.mean(axis=1), x.std(axis=1, ddof=1), out=np.zeros(len(x)), where=x.std(axis=1, ddof=1)>1e-14)*np.sqrt(252)
        sy = np.divide(y.mean(axis=1), y.std(axis=1, ddof=1), out=np.zeros(len(y)), where=y.std(axis=1, ddof=1)>1e-14)*np.sqrt(252)
        sharpe_delta.extend(sx-sy)
    return {"block_days": block_days, "draws": draws,
        "return_positive_probability": float(np.mean(np.array(annual_delta)>0)),
        "sharpe_positive_probability": float(np.mean(np.array(sharpe_delta)>0)),
        "annual_return_delta_low": float(np.quantile(annual_delta, .025)),
        "annual_return_delta_high": float(np.quantile(annual_delta, .975)),
        "sharpe_delta_low": float(np.quantile(sharpe_delta, .025)),
        "sharpe_delta_high": float(np.quantile(sharpe_delta, .975))}


def evaluate(dailies, folds, manifests, root=ROOT, draws=3000):
    summaries, boots, stress = [], [], []
    promotions = {}
    calendars = [tuple(frame.date) for frame in dailies.values()]
    if not calendars or any(c != calendars[0] for c in calendars) or calendars[0][0] != START or calendars[0][-1] != END:
        raise ValueError("Twelve portfolios must have the same full 2022-01-05..2026-07-15 calendar")
    for profile in PROFILES:
        for layer in LAYERS + CONTROL_LAYERS:
            daily = dailies[profile, layer]
            for start, label in [(START, "full"), (pd.Timestamp("2024-01-01"), "2024_plus")]:
                sub = daily.loc[daily.date >= start]
                summaries.append({"profile": profile, "layer": layer, "period": label,
                    "start_date": str(sub.date.min().date()), "end_date": str(sub.date.max().date()), "days": len(sub),
                    **statistics(sub.strategy_return), "mean_daily_turnover": float(sub.turnover.mean()),
                    "total_cost": float(sub.transaction_cost.sum()), "model_used_days": int(sub.model_used.sum()),
                    "active_forecast_days": int(sub.active_components.gt(0).sum()),
                    "mean_target_asset_count": float(sub.target_asset_count.mean()),
                    "min_target_asset_count": int(sub.target_asset_count.min()),
                    "max_target_asset_count": int(sub.target_asset_count.max()),
                    "quote_blocked_orders": int(sub.quote_order_blocked.sum()),
                    "unfinished_risk_correction_days": int(sub.unfinished_risk_correction.sum()),
                    "investable_asset_count": len(manifests[profile]["investable_assets"])})
            for bps in [5, 10, 20]:
                stressed = daily.gross_return - daily.turnover*bps/10000
                stress.append({"profile": profile, "layer": layer, "transaction_cost_bps": bps,
                    **statistics(stressed), "scope": "fixed_decisions_cost_sensitivity_not_retrained_policy"})
        a1 = dailies[profile, "A1"].loc[lambda d: d.date >= "2024-01-01"]
        a3 = dailies[profile, "A3"].loc[lambda d: d.date >= "2024-01-01"]
        for comparison in [("A2", "A1"), ("A3", "A2"), ("A3", "A1"), ("R1", "R0")]:
            p = dailies[profile, comparison[0]].loc[lambda d: d.date >= "2024-01-01"]
            b = dailies[profile, comparison[1]].loc[lambda d: d.date >= "2024-01-01"]
            for length in [20, 60]:
                boots.append({"profile": profile, "comparison": "-".join(comparison),
                    **bootstrap_comparison(p.strategy_return, b.strategy_return, length, draws)})
        ps, bs = statistics(a3.strategy_return), statistics(a1.strategy_return)
        relevant = [b for b in boots if b["profile"] == profile and b["comparison"] == "A3-A1"]
        evidence = (bool(a3.model_used.any()) and bool(a3.active_components.gt(0).any())
                    and any(f.get("deployment_gate_passed", False) for f in folds.get(profile, [])))
        passed = (ps["annual_return"] >= bs["annual_return"] and ps["sharpe"] >= bs["sharpe"]
                  and all(b["return_positive_probability"] >= .8 and b["sharpe_positive_probability"] >= .6 for b in relevant)
                  and evidence)
        promotions[profile] = {"historical_research_promotion_passed": bool(passed),
            "annual_return_delta_vs_A1": ps["annual_return"]-bs["annual_return"],
            "sharpe_delta_vs_A1": ps["sharpe"]-bs["sharpe"],
            "independent_forecast_and_execution_evidence": evidence,
            "a2_a3_same_target": manifests[profile]["a2_a3_same_target"],
            "formal_live_promotion": False, "forward_confirmation": "not_started",
            "missing_items": [] if evidence else ["daily active learned forecast and independent qualified execution evidence"]}
    reports = Path(root) / "output/reports"
    pd.DataFrame(summaries).to_csv(reports / "v45_ablation_summary.csv", index=False)
    pd.DataFrame(boots).to_csv(reports / "v45_bootstrap_robustness.csv", index=False)
    pd.DataFrame(stress).to_csv(reports / "v45_cost_stress.csv", index=False)
    result = {"profiles": promotions, "all_three_historical_research_passed": all(p["historical_research_promotion_passed"] for p in promotions.values()),
        "forward_confirmation": False, "research_only": True, "bootstrap_draws": draws,
        "cost_stress_scope": "fixed decisions; does not claim reoptimized or retrained rewards"}
    write_json(reports / "v45_portfolio_gate.json", result)
    return result


def validate_effective_config(cfg):
    ctl, portfolio = cfg["controller"], cfg["portfolio"]
    if ctl["seeds"] != [17,29,47] or ctl.get("years", [2024,2025,2026]) != [2024,2025,2026]:
        raise ValueError("First-round PPO requires three registered seeds and annual folds")
    if ctl.get("learning_rates", [1e-4,3e-4]) != [1e-4,3e-4] or ctl.get("entropy_weights", [.004,.01]) != [.004,.01]:
        raise ValueError("PPO grid must be the registered four candidates")
    if ctl["max_epochs"] > 120 or ctl.get("teacher_horizon_days", 5) != 5:
        raise ValueError("PPO epoch/teacher contract")
    if portfolio.get("leverage", 1.) != 1 or not portfolio.get("no_cash", True) or portfolio["transaction_cost_bps"] != 5:
        raise ValueError("v45 primary accounting is fully invested/no leverage/5bp")
    if portfolio["max_asset_share_of_sleeve"] != .75:
        raise ValueError("v45 concentration cap must be .75")
    if portfolio.get("anchor_policy", "closest_feasible_anchor_joint_projection_v1") != "closest_feasible_anchor_joint_projection_v1":
        raise ValueError("Unregistered anchor/risk override policy")


def run(cfg, prepare_only=False, train_ppo=False, resume=False, root=ROOT):
    root = Path(root)
    reports = root / "output/reports"
    reports.mkdir(parents=True, exist_ok=True)
    validate_effective_config(cfg)
    write_json(reports / "v45_resolved_portfolio_config.json", {
        "controller": cfg["controller"], "portfolio": cfg["portfolio"], "v45": cfg["v45"],
        "accounting_outputs": {k: cfg["outputs"][k] for k in ["dataset_npz", "metadata"]},
        "mapping": {"controller.minimum_hold_days": "portfolio.minimum_hold_days",
            "portfolio.max_asset_active_weight": "portfolio.asset_active_limits",
            "solver.max_active_deviation": "portfolio.daily_active_limits",
            "solver.max_sleeve_turnover": "portfolio.daily_target_change_limits"},
        "maximum_fits": 108, "training_on_zero_forecast": "allowed_shadow_only",
        "deployment_requires_daily_active_learned_increment": True,
        "fixed_forecast_horizon_weights": HORIZONS,
        "target_cache_independent_of_PPO_config_and_reward_code": True,
        "A0_definition": "zero_mu_constrained_covariance_risk_budget_control_not_unmodified_risk_parity",
        "R0_R1_definition": "daily_confidence_counterfactual_on_common_frozen_monthly_anchor",
        "execution_policy": "block_whole_order_if_quote_rescaling_creates_or_worsens_hard_book_violation"})
    stage = {"status": "running", "engineering_passed": False, "training_requested": train_ppo,
        "full_training_launched_by_this_runner": train_ppo, "missing_items": []}
    stage_path = reports / "v45_portfolio_stage_gate.json"
    write_json(stage_path, stage)
    try:
        returns, modules, quotes, gaps = frozen_accounting(cfg, root)
        gaps.to_csv(reports / "v45_execution_valuation_gaps.csv", index=False)
        predictions, forecast_status = load_predictions(cfg, root)
        manifests = {}
        for profile in PROFILES:
            manifests[profile] = prepare_profile(profile, cfg, returns, quotes, predictions, forecast_status, resume, root)
        if len({m["calendar_hash"] for m in manifests.values()}) != 1:
            raise ValueError("Three target calendars differ")
        if prepare_only:
            stage.update(status="targets_prepared", engineering_passed=False, target_contract_passed=True, forecast=forecast_status,
                missing_items=forecast_status["missing_items"], manifests=manifests,
                execution_validation="pending", promotion=False)
            write_json(stage_path, stage)
            return stage
        dailies, folds, envs = {}, {}, {}
        # Validate all mechanical execution paths before any long PPO fitting.
        for profile in PROFILES:
            folds[profile] = []
            for layer in ["A0", "A1", "A2", "R0", "R1"]:
                env = load_environment(profile, layer, cfg, returns, quotes, root)
                daily, holdings = replay(env)
                dailies[profile, layer] = daily
                daily.to_csv(output(profile, layer, "daily", root), index=False)
                holdings.to_parquet(output(profile, layer, "holdings", root), index=False)
                if layer == "A2":
                    envs[profile] = env
        for profile in PROFILES:
            env, models = envs[profile], {}
            for year in cfg["controller"].get("years", [2024,2025,2026]):
                if train_ppo:
                    approved, fold = train_fold(env, year, resume, root)
                    models[year] = approved
                else:
                    fold = {"year": year, "reason": "no_training_requested", "shadow": True}
                    models[year] = []
                folds[profile].append(fold)
            daily, holdings = replay(env, models)
            dailies[profile, "A3"] = daily
            daily.to_csv(output(profile, "A3", "daily", root), index=False)
            holdings.to_parquet(output(profile, "A3", "holdings", root), index=False)
        result = evaluate(dailies, folds, manifests, root)
        stage.update(status="completed_research", engineering_passed=True, forecast=forecast_status,
            missing_items=forecast_status["missing_items"], folds=folds,
            target_manifests={p: m["signature"] for p,m in manifests.items()},
            promotion=result, research_only=True)
    except Exception as exc:
        stage.update(status="blocked", reason=str(exc), exception_type=type(exc).__name__,
            detail=getattr(exc, "detail", {}), missing_items=[str(exc)])
        write_json(stage_path, stage)
        raise
    write_json(stage_path, stage)
    return stage
