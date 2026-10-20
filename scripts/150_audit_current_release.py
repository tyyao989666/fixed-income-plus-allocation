from __future__ import annotations

"""Audit the frozen no-Agent release before sharing or rerunning it."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, passed: bool, evidence: str, severity: str = "error") -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    release_path = ROOT / "config" / "current_release.yaml"
    release = _load_yaml(release_path)
    cutoff = pd.Timestamp(release["data_cutoffs"]["daily_complete_as_of"])
    monthly_cutoff = pd.Timestamp(release["data_cutoffs"]["monthly_complete_as_of"])
    processed = ROOT / "data_processed"
    reports = ROOT / "output" / "reports"
    checks: list[dict[str, object]] = []

    raw_path = processed / "asset_returns_daily.parquet"
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    last_by_asset = raw.groupby("asset_code")["date"].max()
    common_last = last_by_asset.min()
    checks.append(_check(
        "daily_raw_cross_section_cutoff",
        common_last == cutoff,
        f"configured={cutoff.date()}, common_last={common_last.date()}, assets={len(last_by_asset)}",
    ))

    monthly = pd.read_parquet(processed / "asset_returns_monthly.parquet")
    monthly["date"] = pd.to_datetime(monthly["date"])
    checks.append(_check(
        "monthly_raw_cutoff",
        monthly["date"].max() == monthly_cutoff,
        f"configured={monthly_cutoff.date()}, observed={monthly['date'].max().date()}",
    ))

    market_cfg = _load_yaml(ROOT / "config" / "market_skill_config.yaml")
    canonical = market_cfg.get("canonical_assets", {}).get("rate_30y", {})
    checks.append(_check(
        "market_state_30y_canonical_definition",
        canonical.get("canonical_code") == "rate_30y_curve_proxy" and float(canonical.get("modified_duration", 0)) == 18.5,
        json.dumps(canonical, ensure_ascii=False),
    ))
    checks.append(_check(
        "market_state_cutoff",
        pd.Timestamp(market_cfg.get("release", {}).get("daily_as_of_date")) == cutoff,
        f"configured={market_cfg.get('release', {}).get('daily_as_of_date')}",
    ))

    strategy_cfg = _load_yaml(ROOT / "config" / "v1_strategy_config.yaml")
    strategy_proxy = strategy_cfg.get("synthetic_assets", {}).get("rate_30y_curve_proxy", {})
    checks.append(_check(
        "formal_portfolio_30y_canonical_definition",
        float(strategy_proxy.get("modified_duration", 0)) == 18.5,
        json.dumps(strategy_proxy, ensure_ascii=False),
    ))

    target_path = processed / "v1_balanced_target_weights_monthly.parquet"
    target = pd.read_parquet(target_path)
    target_assets = set(target["asset_code"].astype(str)) if "asset_code" in target.columns else set()
    checks.append(_check(
        "formal_weights_use_curve_proxy",
        "rate_30y_curve_proxy" in target_assets and "rate_30y" not in target_assets,
        f"contains_curve_proxy={'rate_30y_curve_proxy' in target_assets}, contains_raw={'rate_30y' in target_assets}",
    ))

    cards_path = processed / "market_state_cards_daily_v1.parquet"
    formal_metrics: list[dict[str, object]] = []
    for profile in ["low_vol", "balanced", "high_vol"]:
        metric_path = reports / f"v1_{profile}_backtest_metrics.csv"
        metric_frame = pd.read_csv(metric_path)
        values = dict(zip(metric_frame["metric"], metric_frame["value"]))
        formal_metrics.append(
            {
                "profile": profile,
                "annual_return": float(values["annual_return"]),
                "annual_vol": float(values["annual_vol"]),
                "sharpe": float(values["sharpe"]),
                "max_drawdown": float(values["max_drawdown"]),
                "avg_turnover": float(values["avg_turnover"]),
            }
        )
        checks.append(_check(
            f"{profile}_formal_output_refreshed_after_cards",
            metric_path.stat().st_mtime >= cards_path.stat().st_mtime,
            f"metrics_mtime={pd.Timestamp(metric_path.stat().st_mtime, unit='s')}, cards_mtime={pd.Timestamp(cards_path.stat().st_mtime, unit='s')}",
        ))

    gate = json.loads((reports / "hierarchical_asset_itransformer_v14_residual_utility_gate.json").read_text(encoding="utf-8"))
    checks.append(_check(
        "itransformer_direct_alpha_gate_is_off",
        not bool(gate.get("enabled_for_portfolio_stage", gate.get("enabled", False))),
        f"enabled={gate.get('enabled_for_portfolio_stage', gate.get('enabled'))}, sleeve_rank_ic={gate.get('summary', {}).get('sleeve_rank_ic')}",
        severity="warning",
    ))

    robust_cfg = _load_yaml(ROOT / "config" / "robust_execution_ppo_v1_config.yaml")
    checks.append(_check(
        "robust_controller_uses_release_cutoff",
        pd.Timestamp(robust_cfg["data"].get("complete_data_as_of")) == cutoff,
        f"configured={robust_cfg['data'].get('complete_data_as_of')}",
    ))

    rows = pd.DataFrame(checks)
    reports.mkdir(parents=True, exist_ok=True)
    rows.to_csv(reports / "current_release_audit.csv", index=False, encoding="utf-8-sig")

    artifacts = [
        release_path,
        ROOT / "config" / "market_skill_config.yaml",
        ROOT / "config" / "v1_strategy_config.yaml",
        ROOT / "config" / "robust_execution_ppo_v1_config.yaml",
        raw_path,
        processed / "asset_returns_monthly.parquet",
        target_path,
        cards_path,
        processed / "market_skill_views_mechanical_daily_v1.parquet",
        processed / "ep_views_daily_v1.parquet",
        *[reports / f"v1_{profile}_backtest_metrics.csv" for profile in ["low_vol", "balanced", "high_vol"]],
    ]
    manifest = {
        "release": release["version"],
        "data_cutoffs": release["data_cutoffs"],
        "artifacts": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": _hash(path)}
            for path in artifacts if path.exists()
        ],
    }
    (reports / "current_release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# 当前正式无 Agent 版本审计",
        "",
        f"- release: `{release['version']}`",
        f"- 日频共同数据截止: `{cutoff.date()}`",
        f"- 月频数据截止: `{monthly_cutoff.date()}`",
        f"- 通过: `{int(rows['passed'].sum())}/{len(rows)}`",
        "",
        rows.to_markdown(index=False),
        "",
        "## 同步后的正式月频基线",
        "",
        pd.DataFrame(formal_metrics).to_markdown(index=False, floatfmt=".4f"),
        "",
        "说明：Robust PPO 是门控研究 challenger；本审计不把其历史回放结果当作正式基线收益。",
    ]
    (reports / "current_release_audit.md").write_text("\n".join(markdown), encoding="utf-8")
    if not bool(rows["passed"].all()):
        failed = rows.loc[~rows["passed"], "check"].tolist()
        raise SystemExit(f"release audit failed: {failed}")
    print(json.dumps({"passed": int(rows['passed'].sum()), "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
