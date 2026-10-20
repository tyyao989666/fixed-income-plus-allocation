"""Offline v45 forecast data contract; no strategy imports or network access.

All timestamps are int64 nanoseconds. Zero is a tensor placeholder ONLY where
the corresponding mask is false. These 38 context tokens are not a trade list.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PREFIX = "hierarchical_asset_itransformer_v43_clean_calendar"
MODULES = ["rates", "credit", "convertible_bonds", "equity", "alternatives"]
HORIZONS = [5, 20, 60]
NAT = np.iinfo(np.int64).min
GRID_NAMES = [f"bond_{b}_equity_{e}" for b in ("strong", "neutral", "weak")
              for e in ("strong", "neutral", "weak")]
LABEL_REFERENCE_VERSION = "fixed_economic_calendar_core_v2"


def label_reference_policy(assets, modules):
    """Parent-approved fixed research comparator, NOT an investment universe.

    Definitions follow economic sleeves/calendars, not future coverage, scores,
    or returns. Every modeled asset retains its individual supervised target.
    This synthetic comparator is not a certified wealth/total-return index.
    """
    members = {
        "rates": ["rate_1_3y", "rate_3_5y", "rate_7_10y", "rate_30y_curve_proxy"],
        "credit": ["credit_ind_aaa"],
        "convertible_bonds": [f"cb_{style}_{size}_{price}" for style in ("balanced", "bond", "equity")
                              for size in ("large", "small") for price in ("high", "low")],
        "equity": ["equity_broad", "equity_csi1000"],
        "alternatives": ["alt_gold", "alt_commodity", "alt_nonferrous", "alt_chemical"],
    }
    policy = {module: {code: 1.0 / len(codes) for code in codes} for module, codes in members.items()}
    preferred = np.zeros((5, len(assets)), dtype=float)
    for s, module in enumerate(MODULES):
        for code, weight in policy[module].items():
            if code not in assets or modules[assets.index(code)] != module:
                raise ValueError(f"missing or wrong-sleeve research reference member: {code}")
            preferred[s, assets.index(code)] = weight
    return preferred, policy


def annual_splits(arrays: Mapping, year: int) -> dict:
    """T -> 60 obs purge -> V126 -> 60 obs purge -> G126.

    The longest label must mature no later than the last observation before
    January 1 of year. G is never refitted into T. Insufficient T is returned
    with a nonempty reason, not repaired by shortening purges. Accepts NpzFile
    or an ordinary dict. Validity is per component, never all-38 intersection.
    """
    dates = np.asarray(arrays["dates"], dtype=np.int64)
    ends = np.asarray(arrays["label_end"], dtype=np.int64)
    if len(dates) == 0 or np.any(np.diff(dates) <= 0):
        raise ValueError("dates must be nonempty, unique and increasing")
    if ends.shape != (len(dates), 3):
        raise ValueError("label_end must have shape [T,3] for horizons 5/20/60")
    jan = pd.Timestamp(year=year, month=1, day=1).value
    next_jan = pd.Timestamp(year=year + 1, month=1, day=1).value
    past = np.flatnonzero(dates < jan)
    cutoff = int(dates[past[-1]]) if len(past) else NAT
    samples = np.asarray(arrays.get("sample_indices", np.arange(119, len(dates))), dtype=np.int64)
    if np.any((samples < 119) | (samples >= len(dates))) or np.any(np.diff(samples) <= 0):
        raise ValueError("sample_indices must be increasing and >=119")
    test = samples[(dates[samples] >= jan) & (dates[samples] < next_jan)]
    mature = samples[(dates[samples] <= cutoff) & (ends[samples, -1] != NAT)
                     & (ends[samples, -1] <= cutoff)]
    result = dict(train=np.array([], dtype=np.int64), validation=np.array([], dtype=np.int64),
                  gate=np.array([], dtype=np.int64), test=test, reason="", cutoff=cutoff)
    if len(mature) < 372:
        result["reason"] = "insufficient_mature_history_for_V126_G126_and_two_60_observation_purges"
        return result
    # Boundaries use original observation positions, not compressed valid cells.
    gate_stop = int(mature[-1]) + 1
    gate_start = gate_stop - 126
    val_stop = gate_start - 60
    val_start = val_stop - 126
    train_stop = val_start - 60
    valid = np.ones(len(dates), dtype=bool)
    if "asset_target_mask" in arrays:
        valid = np.asarray(arrays["asset_target_mask"])[:, -1].astype(bool).any(axis=1)
        if "sleeve_target_mask" in arrays:
            valid |= np.asarray(arrays["sleeve_target_mask"])[:, -1].astype(bool).any(axis=1)
    usable = mature[valid[mature]]
    result["train"] = usable[usable < train_stop]
    result["validation"] = usable[(usable >= val_start) & (usable < val_stop)]
    result["gate"] = usable[(usable >= gate_start) & (usable < gate_stop)]
    if len(result["train"]) < 300:
        result["reason"] = "insufficient_train_dates_minimum_300"
    elif len(result["validation"]) != 126 or len(result["gate"]) != 126:
        result["reason"] = "insufficient_valid_validation_or_gate_dates_need_126_each"
    return result


def robust_scale(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """At t use x[t], but location/scale only x[t-252:t], minimum126.

    No global clipping/fitting. Constant history uses an explicit 1e-8 IQR floor.
    A raw observation without enough scaler history is not an available feature.
    """
    frame = frame.replace([np.inf, -np.inf], np.nan)
    past = frame.shift(1).rolling(252, min_periods=126)
    median = past.median()
    iqr = (past.quantile(0.75) - past.quantile(0.25)).clip(lower=1e-8)
    scaled = (frame - median) / iqr
    mask = np.isfinite(scaled.to_numpy())
    return np.where(mask, scaled.to_numpy(), 0).astype(np.float32), mask


def local_features(returns: pd.DataFrame, ohlcv: dict[str, pd.DataFrame]):
    """Observed returns and source-filtered local bars only; no fill across gaps.

    Features are known at close t, executed at t+1. Every rolling return feature
    needs a completely observed window. No fake prices/volumes for curve tokens.
    """
    raw = {"ret_1d": returns}
    log_returns = np.log1p(returns)
    for window in (5, 20, 60, 120, 252):
        raw[f"mom_{window}d"] = np.expm1(log_returns.rolling(window, min_periods=window).sum())
    for window in (20, 60, 120, 252):
        raw[f"vol_{window}d"] = returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252)
    # Ratios computed within each window cannot carry an artificial price through a gap.
    def ma_deviation(x):
        prices = np.cumprod(1 + x)
        return prices[-1] / prices.mean() - 1

    def drawdown(x):
        prices = np.r_[1.0, np.cumprod(1 + x)]
        return np.min(prices / np.maximum.accumulate(prices) - 1)

    for window in (20, 60):
        raw[f"ma_deviation_{window}d"] = returns.rolling(window, min_periods=window).apply(ma_deviation, raw=True)
    raw["dd_60d"] = returns.rolling(60, min_periods=60).apply(drawdown, raw=True)
    op, hi, lo, cl = (ohlcv[k] for k in ("open", "high", "low", "close"))
    raw["overnight_ret"] = op / cl.shift(1) - 1
    raw["intraday_ret"] = cl / op - 1
    raw["range_pct"] = hi / lo - 1
    raw["close_position"] = ((cl - lo) / (hi - lo).where(hi > lo)) - 0.5
    for column in ("vol", "amount"):
        positive = ohlcv[column].where(ohlcv[column] > 0)
        raw[f"log_{column}"] = np.log1p(positive)
        raw[f"{column}_change_1d"] = positive / positive.shift(1) - 1
    values, masks = zip(*(robust_scale(frame) for frame in raw.values()))
    return np.stack(values, axis=-1), np.stack(masks, axis=-1), list(raw)


def reference_weights(returns, return_masks, preferred, module_index):
    """Freeze membership using ONLY observed t, then freeze weights through h.

    A prelisting token receives no weight. This is not future-window coverage
    reweighting. If no preferred member exists the reference is unavailable.
    """
    observed = np.array(return_masks, dtype=bool, copy=True)
    observed &= np.isfinite(returns)
    weights = observed[:, None, :] * np.asarray(preferred)[None, :, :]
    for s in range(5):
        weights[:, s, np.asarray(module_index) != s] = 0
    totals = weights.sum(axis=-1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights, dtype=float), where=totals > 0)


def relative_labels(dates, returns, return_masks, module_index, sleeve_weights, balanced_weights):
    """Raw h-day returns t+1..t+h, buy-and-hold references (natural drift).

    The exact endpoint formula sum_i w_i(t)*prod_d(1+r_i,d)-1 is equivalent to
    daily drifting holdings, NOT a daily constant-weight portfolio. A missing
    required future observation invalidates that reference and related labels.
    """
    dates = np.asarray(dates, dtype=np.int64)
    r = np.asarray(returns, dtype=float)
    observed = np.asarray(return_masks, dtype=bool) & np.isfinite(r) & (r > -1)
    t_count, a_count = r.shape
    sw = np.asarray(sleeve_weights, dtype=float)
    bw = np.asarray(balanced_weights, dtype=float)
    if sw.shape != (t_count, 5, a_count) or bw.shape != (t_count, 5):
        raise ValueError("reference weights must have shape [T,5,A] and [T,5]")
    if np.any(sw < 0) or np.any(bw < 0):
        raise ValueError("negative reference weights")
    for weights in (sw, bw):
        totals = weights.sum(axis=-1)
        if not np.all(np.isclose(totals, 1, atol=1e-8) | np.isclose(totals, 0, atol=1e-8)):
            raise ValueError("reference weights must sum to one or zero (unavailable)")
    asset = np.zeros((t_count, 3, a_count), dtype=np.float32)
    asset_mask = np.zeros_like(asset, dtype=bool)
    sleeve = np.zeros((t_count, 3, 5), dtype=np.float32)
    sleeve_mask = np.zeros_like(sleeve, dtype=bool)
    ends = np.full((t_count, 3), NAT, dtype=np.int64)
    for t in range(t_count):
        for j, h in enumerate(HORIZONS):
            if t + h >= t_count:
                continue
            ends[t, j] = dates[t + h]
            future_mask = observed[t + 1:t + h + 1].all(axis=0)
            growth = np.prod(1 + np.where(observed[t + 1:t + h + 1], r[t + 1:t + h + 1], 0), axis=0)
            ref_valid = (sw[t].sum(axis=1) > 0) & ((sw[t] == 0) | future_mask[None, :]).all(axis=1)
            ref = sw[t] @ (growth - 1)
            valid = observed[t] & future_mask & ref_valid[np.asarray(module_index)]
            asset[t, j] = np.where(valid, growth - 1 - ref[np.asarray(module_index)], 0)
            asset_mask[t, j] = valid
            common_valid = bw[t].sum() > 0 and np.all((bw[t] == 0) | ref_valid)
            valid_sleeve = ref_valid & common_valid
            sleeve[t, j] = np.where(valid_sleeve, ref - bw[t] @ ref, 0)
            sleeve_mask[t, j] = valid_sleeve
    return dict(asset_target=asset, asset_target_mask=asset_mask, sleeve_target=sleeve,
                sleeve_target_mask=sleeve_mask, label_end=ends)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_bars(raw, dates, assets, modules):
    """Allow exchange/index bars only; credit/curve/bucket pseudo-bars excluded."""
    source = raw.data_source.fillna("").astype(str)
    allowed = source.str.startswith(("tushare", "wind_underlying_index", "wind_nanhua:"))
    allowed &= raw.module.isin(["equity", "alternatives"])
    if "data_quality_flag" in raw:
        allowed &= raw.data_quality_flag.eq("trading_day_observation")
    bars = {}
    for field in ("open", "high", "low", "close", "vol", "amount"):
        frame = raw.loc[allowed, ["date", "asset_code", field]].copy()
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        bars[field] = frame.pivot(index="date", columns="asset_code", values=field).reindex(index=dates, columns=assets)
        bars[field] = bars[field].where(bars[field] > 0)
    good = (bars["high"] >= bars["low"]) & (bars["open"] >= bars["low"]) & (bars["open"] <= bars["high"])
    good &= (bars["close"] >= bars["low"]) & (bars["close"] <= bars["high"])
    for field in ("open", "high", "low", "close"):
        bars[field] = bars[field].where(good)
    return bars


def observable_grid(returns: pd.DataFrame):
    """Nine-state geometry using price evidence only, not the old macro rule.

    Fixed research definition, no fitted/full-sample quantiles: 20-observation
    compounded return, rates 7-10Y +/-0.1%, broad equity +/-0.5%. Unknown if any
    return is missing. The observation at t is known at close t. This deliberately
    does NOT certify the original frozen cards' macro-informed state as PIT.
    """
    states, scores, score_masks = [], [], []
    for asset, threshold in (("rate_7_10y", 0.001), ("equity_broad", 0.005)):
        momentum = np.expm1(np.log1p(returns[asset]).rolling(20, min_periods=20).sum())
        state = pd.Series("neutral", index=returns.index)
        state.loc[momentum > threshold] = "strong"
        state.loc[momentum < -threshold] = "weak"
        states.append(state.where(momentum.notna()))
        available = momentum.notna().to_numpy()
        scores.append(np.where(available, 0.5 + 0.5 * np.tanh(momentum.to_numpy() / threshold), 0).astype(np.float32))
        score_masks.append(available)
    regime = "bond_" + states[0] + "_equity_" + states[1]
    known = regime.isin(GRID_NAMES).to_numpy()
    values = np.column_stack([regime.eq(n).fillna(False).to_numpy(dtype=np.float32) for n in GRID_NAMES])
    index = pd.Categorical(regime, categories=GRID_NAMES).codes.astype(np.int64)
    values = np.column_stack([values, *scores])
    masks = np.column_stack([np.repeat(known[:, None], 9, axis=1), *score_masks])
    return values, masks, index


def build_dataset(root: Path = ROOT) -> tuple[dict, dict, pd.DataFrame]:
    """Read frozen NPZ plus existing on-disk sources; return arrays/metadata/audit."""
    root = Path(root)
    paths = dict(frozen_npz=root / f"data_processed/{FROZEN_PREFIX}_dataset.npz",
                 frozen_metadata=root / f"output/reports/{FROZEN_PREFIX}_metadata.json",
                 local_bars=root / "data_processed/asset_returns_daily.parquet",
                 cards=root / "data_processed/v44_state_cards_daily.parquet",
                 balanced_reference=root / "data_processed/v44_balanced_risk_budget_base_monthly.parquet",
                 reference_implementation=root / "src/v45_data.py")
    old_meta = json.loads(paths["frozen_metadata"].read_text(encoding="utf-8"))
    with np.load(paths["frozen_npz"], allow_pickle=False) as frozen:
        arrays = {k: frozen[k].copy() for k in ("dates", "returns", "return_masks", "module_index")}
    dates = pd.DatetimeIndex(pd.to_datetime(arrays["dates"], unit="ns"))
    assets = old_meta["assets"]
    modules = old_meta["asset_modules"]
    if len(dates) != 2069 or dates[0] != pd.Timestamp("2018-01-02") or dates[-1] != pd.Timestamp("2026-07-15"):
        raise ValueError("unexpected frozen calendar; do not silently refresh v45")
    if len(assets) != 38 or len(set(assets)) != 38 or arrays["returns"].shape != (2069, 38):
        raise ValueError("expected frozen 38-token universe")
    if not np.array_equal(arrays["module_index"], [MODULES.index(m) for m in modules]):
        raise ValueError("module_index does not agree with frozen metadata")
    if np.any(np.diff(arrays["dates"]) <= 0):
        raise ValueError("frozen calendar must be ordered and unique")
    observed = arrays["return_masks"].astype(bool)
    if not np.isin(arrays["return_masks"], [0, 1]).all():
        raise ValueError("return masks must be binary")
    if not np.isfinite(arrays["returns"][observed]).all() or (arrays["returns"][observed] <= -1).any():
        raise ValueError("invalid observed frozen return")
    returns = pd.DataFrame(np.where(observed, arrays["returns"], np.nan), index=dates, columns=assets)
    raw = pd.read_parquet(paths["local_bars"])
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw.loc[raw.date.isin(dates) & raw.asset_code.isin(assets)].copy()
    if raw.duplicated(["date", "asset_code"]).any():
        raise ValueError("ambiguous duplicate source bars")
    bars = _source_bars(raw, dates, assets, modules)
    arrays["local_values"], arrays["local_masks"], names = local_features(returns, bars)
    cards = pd.read_parquet(paths["cards"], columns=["date", "stock_bond_regime"])
    cards["date"] = pd.to_datetime(cards.date)
    regime = cards.set_index("date").reindex(dates).stock_bond_regime
    arrays["frozen_card_regime_index"] = pd.Categorical(regime, categories=GRID_NAMES).codes.astype(np.int64)
    arrays["macro_values"], arrays["macro_masks"], arrays["regime_index"] = observable_grid(returns)
    preferred, preferred_dict = label_reference_policy(assets, modules)
    sw = reference_weights(arrays["returns"], arrays["return_masks"], preferred, arrays["module_index"])
    anchor = pd.read_parquet(paths["balanced_reference"])
    anchor["date"] = pd.to_datetime(anchor.date)
    if anchor.date.duplicated().any():
        raise ValueError("duplicate balanced reference dates")
    aligned = pd.merge_asof(pd.DataFrame({"date": dates}), anchor.sort_values("date"), on="date", direction="backward")
    bw = aligned[MODULES].to_numpy(dtype=float)
    available = np.isfinite(bw).all(axis=1)
    if np.any(bw[available] < 0) or not np.allclose(bw[available].sum(axis=1), 1, atol=1e-8):
        raise ValueError("balanced reference must be a no-cash unit simplex")
    bw[~available] = 0
    arrays["sleeve_reference_weights"] = sw
    arrays["balanced_reference_weights"] = bw
    arrays.update(relative_labels(arrays["dates"], arrays["returns"], arrays["return_masks"], arrays["module_index"], sw, bw))
    arrays["sample_indices"] = np.arange(119, len(dates), dtype=np.int64)
    arrays["horizons_days"] = np.asarray(HORIZONS, dtype=np.int64)
    metadata = dict(version="v45_strict_data", rows=len(dates), start=str(dates[0].date()), end=str(dates[-1].date()),
                    assets=assets, asset_names=old_meta["asset_names"], asset_modules=modules, modules=MODULES,
                    horizons_days=HORIZONS, local_features=names,
                    macro_features=[f"grid_{n}" for n in GRID_NAMES] + ["grid_bond_score", "grid_equity_score"],
                    sequence_length_days=120, samples=len(arrays["sample_indices"]),
                    normalization=dict(window=252, min_periods=126, statistics_shift=1, method="median_IQR", iqr_floor=1e-8,
                                       raw_feature_cutoff="close_t", execution="next_observation_t_plus_1", clipping="none"),
                    context_universe_note="38 model context tokens only; neither product investability nor actual holdings implied",
                    macro_policy="No raw macro/valuation or original card-state inputs. Nine-grid derived only from observed frozen returns; original source-rule PIT NOT fully audited.",
                    grid_definition=dict(source="frozen_return_masks_and_returns", window=20,
                                         bond_asset="rate_7_10y", bond_neutral_band=[-0.001, 0.001],
                                         equity_asset="equity_broad", equity_neutral_band=[-0.005, 0.005],
                                         cutoff="known_close_t", missing="unknown onehot: nine zeros with false masks; each continuous axis has its own mask",
                                         continuous_scores="0.5 + 0.5*tanh(compounded_20d_return / axis_neutral_threshold); bond then equity; no fitted thresholds",
                                         continuous_score_units="bounded_0_1_price_evidence_not_original_card_score_or_probability",
                                         distinction="same nine-state geometry, NOT original macro-informed card rules",
                                         frozen_card_regime_index="diagnostic only; unverified PIT; must not enter strict model"),
                    source_rule_pit_fully_audited=False, unverified_macro_excluded=True,
                    label_policy=dict(version=LABEL_REFERENCE_VERSION,
                                      reference_role="fixed label research benchmark ONLY; NOT investment universe, actual holdings, or certified wealth index",
                                      selection_basis="parent-approved economic sleeve/calendar definitions; no full-sample coverage filter, future scores, or return fitting",
                                      change_from_v1="replace product-preferred constituent mix with fixed economic core; not a missing-return fill or relaxed gate",
                                      units="raw_unannualized_h_day_return", horizon="t+1..t+h",
                                      asset_reference="fixed economic-calendar core v2; observed-at-t membership; equal weights renormalized once at t",
                                      member_selection="return_masks[t] only, no future availability or hindsight selection",
                                      preferred_weights=preferred_dict,
                                      sleeve_reference="latest frozen balanced raw risk-budget anchor dated <=t; no pre-anchor backfill",
                                      holding_policy="freeze weights at t; buy-and-hold natural drift; no daily rebalance",
                                      gaps="every required constituent return must be observed on every future observation; never fill",
                                      baseline="not yet residualized; F0 OOF is forecast-stage responsibility",
                                      benchmark_historical_pit="fixed research definition; historical investability and source release-vintage PIT not asserted"),
                    return_policy="bitwise frozen v43 returns and original frozen observation masks; curve/index/bucket research returns, not certified total returns",
                    component_training_policy="annual_splits counts dates with any valid longest-horizon label; each forecast component must separately enforce minimum 300 valid training dates",
                    limitations=["30Y fixed-duration curve proxy, not ETF execution prices",
                                 "Credit membership/duration/wealth and CB historical constituents/calls require independent audit",
                                 "Nanhua indices are not frictionless tradable ETFs",
                                 "REITs prelisting missing remains missing; distribution/total return not certified",
                                 "US duration proxy currency/FX treatment not verified; not certified RMB returns",
                                 "Source prices/volumes and return adjustment conventions not certified as investable total-return series"],
                    excluded_macro_features=old_meta["macro_features"],
                    source_manifest={k: dict(path=str(p.relative_to(root)), sha256=_sha256(p)) for k, p in paths.items()},
                    array_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()})
    audit = field_audit(raw, arrays, metadata, dates)
    return arrays, metadata, audit


def field_audit(raw, arrays, metadata, dates):
    """Field/asset audit, including exclusions and candid source limitations."""
    records = []
    common = dict(frequency="local_trading_observation_daily", calendar="frozen_equity_broad_2069",
                  revision_policy="local snapshot hash; historical release vintages not independently audited",
                  observation_date=f"{dates[0].date()}..{dates[-1].date()}", available_at="assumed known close_t; timestamps not independently certified",
                  missing_policy="no ffill/bfill/prelisting fill; tensor placeholder zero with false mask")
    for a, code in enumerate(metadata["assets"]):
        part = raw.loc[raw.asset_code.eq(code)]
        source = " | ".join(sorted(part.data_source.dropna().astype(str).unique()))
        if code == "rate_30y_curve_proxy":
            source = "frozen v43 fixed-duration treasury_y30 curve proxy"
        module = metadata["asset_modules"][a]
        for field, mask in [("return_1d", arrays["return_masks"][:, a])]+[
                (n, arrays["local_masks"][:, a, f]) for f, n in enumerate(metadata["local_features"])]:
            idx = np.flatnonzero(mask)
            records.append(dict(common, field_id=field, asset_code=code, module=module, source=source,
                                unit="decimal_return" if field == "return_1d" else "trailing_median_IQR_units",
                                currency="unverified_FX" if code == "alt_us_bond" else "source_native_not_certified",
                                proxy_flag=True, feature_role="label_source" if field == "return_1d" else "strict_local",
                                validation_status="frozen_observation_mask_preserved" if field == "return_1d" else "causal_transform_source_semantics_not_full_PIT_certification",
                                observed_count=int(len(idx)), first_valid=str(dates[idx[0]].date()) if len(idx) else None,
                                last_valid=str(dates[idx[-1]].date()) if len(idx) else None))
    for f, name in enumerate(metadata["macro_features"]):
        records.append(dict(common, field_id=name, asset_code="GLOBAL", module="nine_grid", source="frozen_return_masks_and_returns/observable_grid",
                            unit="one_hot" if f < 9 else "bounded_0_1_return_derived_score", currency="not_applicable", proxy_flag=True, feature_role="strict_observable_grid",
                            validation_status="past_observed_return_derivation_verified_NOT_original_card_rules", observed_count=int(arrays["macro_masks"][:, f].sum())))
    records.append(dict(common, field_id="frozen_card_regime_index", asset_code="GLOBAL", module="nine_grid",
                        source="v44_state_cards_daily.parquet/stock_bond_regime", unit="category_index", currency="not_applicable",
                        proxy_flag=True, feature_role="diagnostic_only_excluded_from_strict_inputs",
                        validation_status="source_rule_historical_PIT_NOT_proved", observed_count=int((arrays["frozen_card_regime_index"] >= 0).sum())))
    # The exclusion list is supplied by build/export via metadata, not inferred as approved.
    for name in metadata.get("excluded_macro_features", []):
        records.append(dict(common, field_id=name, asset_code="GLOBAL", module="macro", source=f"{FROZEN_PREFIX}_metadata.json",
                            unit="excluded", currency="unverified", proxy_flag=True, feature_role="excluded_strict_branch",
                            validation_status="excluded_unverified_release_or_revision_PIT", observed_count=0))
    return pd.DataFrame(records)


def export_dataset(root: Path = ROOT, output: Path | None = None, reports: Path | None = None) -> dict:
    root = Path(root)
    output = Path(output) if output else root / "data_processed/v45_dataset.npz"
    reports = Path(reports) if reports else root / "output/reports"
    arrays, metadata, audit = build_dataset(root)
    split_rows = []
    dates = pd.DatetimeIndex(pd.to_datetime(arrays["dates"]))
    for year in range(2022, 2027):
        split = annual_splits(arrays, year)
        row = dict(year=year, reason=split["reason"], cutoff=str(pd.Timestamp(split["cutoff"]).date()))
        for name in ("train", "validation", "gate", "test"):
            idx = split[name]
            row.update({f"{name}_n": len(idx), f"{name}_start": str(dates[idx[0]].date()) if len(idx) else None,
                        f"{name}_end": str(dates[idx[-1]].date()) if len(idx) else None})
        split_rows.append(row)
    metadata["annual_splits"] = split_rows
    coverage = []
    for a, code in enumerate(metadata["assets"]):
        idx = np.flatnonzero(arrays["return_masks"][:, a])
        row = dict(asset_code=code, module=metadata["asset_modules"][a], observations=len(idx),
                   first_valid=str(dates[idx[0]].date()) if len(idx) else None,
                   last_valid=str(dates[idx[-1]].date()) if len(idx) else None,
                   local_valid_cells=int(arrays["local_masks"][:, a].sum()))
        row.update({f"labels_h{h}": int(arrays["asset_target_mask"][:, j, a].sum()) for j, h in enumerate(HORIZONS)})
        coverage.append(row)
    label_coverage = []
    scopes = [("all", None, np.arange(len(dates))) ]
    for year in range(2022, 2027):
        split = annual_splits(arrays, year)
        scopes.extend((stage, year, split[stage]) for stage in ("train", "validation", "gate", "test"))
    for scope, year, indices in scopes:
        for task, components in (("asset", metadata["assets"]), ("sleeve", MODULES)):
            mask = arrays[f"{task}_target_mask"][indices]
            for j, h in enumerate(HORIZONS):
                for c, component in enumerate(components):
                    count = int(mask[:, j, c].sum())
                    label_coverage.append(dict(scope=scope, year=year, task=task, component=component,
                                               horizon=h, valid_dates=count, total_dates=len(indices),
                                               minimum_train_300_met=(count >= 300) if scope == "train" else None))
    output.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata["dataset_sha256"] = _sha256(output)
    (reports / "v45_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    audit.to_csv(reports / "v45_field_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(reports / "v45_data_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(label_coverage).to_csv(reports / "v45_label_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).to_csv(reports / "v45_split_dates.csv", index=False, encoding="utf-8-sig")
    audit.groupby(["field_id", "feature_role", "validation_status"], as_index=False).observed_count.sum().to_csv(
        reports / "v45_features.csv", index=False, encoding="utf-8-sig")
    return metadata
