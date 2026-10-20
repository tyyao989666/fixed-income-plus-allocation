from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


PROFILE_OUTPUTS = {
    "risk_budget_base": "data_processed/v1_{profile}_risk_budget_base_monthly.parquet",
    "sleeve_weights": "data_processed/v1_{profile}_sleeve_weights_monthly.parquet",
    "weights": "data_processed/v1_{profile}_target_weights_monthly.parquet",
    "nav": "output/reports/v1_{profile}_backtest_nav.csv",
    "metrics": "output/reports/v1_{profile}_backtest_metrics.csv",
    "attribution": "output/reports/v1_{profile}_attribution_by_sleeve.csv",
    "robustness": "output/reports/v1_{profile}_robustness_summary.csv",
    "data_quality": "output/reports/v1_{profile}_data_quality.csv",
    "ep_posterior": "data_processed/v1_{profile}_ep_posterior_monthly.parquet",
    "ep_signals": "output/reports/v1_{profile}_ep_signals.csv",
    "optimizer_diagnostics": "output/reports/v1_{profile}_optimizer_diagnostics.csv",
}


def versioned_outputs(version: str) -> dict[str, str]:
    if version == "no_agent":
        return PROFILE_OUTPUTS
    return {key: value.replace("v1_{profile}", f"v1_{version}_{{profile}}") for key, value in PROFILE_OUTPUTS.items()}


def run_command(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def profile_cfg(profile: str, version: str = "no_agent") -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from src.v1_strategy import load_config

    cfg = copy.deepcopy(load_config())
    if profile not in cfg["product"]["profiles"]:
        raise ValueError(f"Unknown profile: {profile}")
    cfg["product"]["selected_profile"] = profile
    cfg["outputs"] = {k: v.format(profile=profile) for k, v in versioned_outputs(version).items()}
    cfg["strategy_version"] = version
    return cfg


def run_profile(profile: str, version: str = "no_agent") -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from src.v1_strategy import (
        build_asset_weights,
        build_risk_budget_base,
        build_sleeve_weights,
        run_attribution,
        run_backtest,
        run_robustness,
    )

    from src.v1_strategy import V1Paths

    cfg = profile_cfg(profile, version)
    paths = V1Paths()
    if version == "agentic":
        profile_views = ROOT / "data_processed" / f"ep_views_agentic_{profile}_daily_v1.parquet"
        paths = V1Paths(views=profile_views if profile_views.exists() else ROOT / "data_processed" / "ep_views_agentic_daily_v1.parquet")
    base = build_risk_budget_base(cfg, paths)
    sleeve = build_sleeve_weights(cfg, paths)
    weights = build_asset_weights(cfg, paths)
    nav, metrics = run_backtest(cfg, paths)
    attribution = run_attribution(cfg, paths)
    robustness = run_robustness(cfg)
    return {
        "profile": profile,
        "strategy_version": version,
        "risk_budget_rows": len(base),
        "sleeve_rows": len(sleeve),
        "weight_rows": len(weights),
        "backtest_months": len(nav),
        "metrics_rows": len(metrics),
        "attribution_rows": len(attribution),
        "robustness_rows": len(robustness),
        "latest_weight_date": str(sleeve["date"].max().date()) if not sleeve.empty else "",
        "nav": cfg["outputs"]["nav"],
        "metrics": cfg["outputs"]["metrics"],
    }


def build_common_synthetic_assets() -> dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from src.v1_strategy import load_config, write_synthetic_assets

    return write_synthetic_assets(load_config())


def parse_profiles(value: str) -> list[str]:
    if value == "all":
        return ["low_vol", "balanced", "high_vol"]
    return [x.strip() for x in value.split(",") if x.strip()]


def write_manifest(results: list[dict[str, Any]], version: str = "no_agent") -> None:
    hybrid_prefix = "agentic_itransformer_ppo" if version == "agentic" else "itransformer_ppo"
    manifest = {
        "pipeline": "v1_full_pipeline",
        "strategy_version": version,
        "profiles": results,
        "common_outputs": {
            "market_state_cards": "data_processed/market_state_cards_daily_v1.parquet",
            "market_skill_views": "data_processed/market_skill_views_daily_v1.parquet",
            "market_skill_mechanical_views": "data_processed/market_skill_views_mechanical_daily_v1.parquet",
            "market_skill_llm_views": "data_processed/market_skill_views_llm_daily_v1.parquet",
            "market_skill_agentic_views": "data_processed/market_skill_views_agentic_daily_v1.parquet",
            "ep_views": "data_processed/ep_views_daily_v1.parquet",
            "agentic_ep_views": "data_processed/ep_views_agentic_daily_v1.parquet",
            "latest_state_card": "output/reports/market_state_card_latest.md",
            "views_report": "output/reports/market_skill_views_report.md",
            "llm_views_report": "output/reports/market_skill_llm_views_report.md",
            "daily_path_itransformer_predictions": "data_processed/itransformer_predictions_daily_path_monthly.parquet",
            "daily_path_itransformer_report": "output/reports/itransformer_daily_validation_report.csv",
            "itransformer_v2_predictions": "data_processed/itransformer_daily_timing_v2_monthly_oos.parquet",
            "itransformer_v2_gate": "output/reports/itransformer_daily_timing_v2_formal_gate.json",
            "itransformer_v2_pit_audit": "output/reports/itransformer_v2_point_in_time_audit.md",
            "itransformer_ppo_summary": f"output/reports/{hybrid_prefix}_three_profile_summary.csv",
            "itransformer_ppo_gate": f"output/reports/{hybrid_prefix}_formal_gate.json",
            "itransformer_ppo_audit": f"output/reports/{hybrid_prefix}_point_in_time_audit.md",
            "itransformer_ppo_report": f"output/reports/{hybrid_prefix}_method_and_results.md",
        },
    }
    name = "v1_full_pipeline_manifest.json" if version == "no_agent" else f"v1_{version}_full_pipeline_manifest.json"
    path = ROOT / "output" / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v1 fixed-income-plus allocation pipeline.")
    parser.add_argument("--profiles", default="all", help="all, or comma-separated profiles: low_vol,balanced,high_vol")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers for product profiles.")
    parser.add_argument("--refresh-latest-data", action="store_true", help="Refresh raw/local data with script 116 before rebuilding v43 panels.")
    parser.add_argument("--refresh-as-of", default=None, help="Optional YYYY-MM-DD as-of date passed to the latest-data refresher.")
    parser.add_argument("--skip-data", action="store_true", help="Skip v43 data rebuild.")
    parser.add_argument("--skip-market-skill", action="store_true", help="Skip MarketSkill state/view rebuild.")
    parser.add_argument("--market-view-mode", choices=["mechanical", "llm", "both"], default=None, help="MarketSkill view generator to run.")
    parser.add_argument("--active-market-views", choices=["mechanical", "llm"], default=None, help="Which generated views become ep_views_daily_v1.")
    parser.add_argument("--llm-view-frequency", choices=["month_end", "latest", "all"], default=None, help="GLM view generation frequency.")
    parser.add_argument("--refresh-daily-itransformer", action="store_true", help="Rebuild and train the daily-path iTransformer signal before allocation.")
    parser.add_argument("--refresh-daily-itransformer-v2", action="store_true", help="Rebuild, walk-forward train, gate, and materialize the identified multi-horizon iTransformer V2 signal.")
    parser.add_argument("--skip-itransformer-ppo", action="store_true", help="Skip the default constrained iTransformer V2 + PPO enhanced-version run.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip V1 PnL/holding/attribution visual report generation.")
    parser.add_argument("--no-parallel", action="store_true", help="Run product profiles sequentially.")
    parser.add_argument("--strategy-version", choices=["no_agent", "agentic", "compare"], default="no_agent")
    parser.add_argument("--agent-frequency", choices=["month_end", "latest", "all"], default="month_end")
    parser.add_argument("--agent-config", default=None, help="Optional YAML config for Agentic Market View generation.")
    args = parser.parse_args()

    if args.refresh_latest_data and args.skip_data:
        parser.error("--refresh-latest-data cannot be combined with --skip-data")

    profiles = parse_profiles(args.profiles)

    if args.refresh_latest_data:
        refresh_args = [PYTHON, "scripts/116_refresh_latest_data.py"]
        if args.refresh_as_of:
            refresh_args.extend(["--as-of", args.refresh_as_of])
        run_command("refresh latest source data", refresh_args)

    if not args.skip_data:
        run_command("rebuild v43 data", [PYTHON, "scripts/47_build_v43_basic_data.py"])
        run_command("audit v43 data", [PYTHON, "scripts/48_audit_v43_no_annual_data.py"])

    if not args.skip_market_skill:
        run_command("generate market state cards", [PYTHON, "scripts/10_generate_market_state_cards.py"])
        view_args = [PYTHON, "scripts/11_run_llm_scenario_engine.py"]
        if args.market_view_mode:
            view_args.extend(["--mode", args.market_view_mode])
        if args.active_market_views:
            view_args.extend(["--active-source", args.active_market_views])
        if args.llm_view_frequency:
            view_args.extend(["--llm-frequency", args.llm_view_frequency])
        run_command("generate MarketSkill views", view_args)
        ep_args = [PYTHON, "scripts/12_build_ep_views.py"]
        if args.active_market_views:
            ep_args.extend(["--source", args.active_market_views])
        run_command("generate EP-ready views", ep_args)
        if args.strategy_version in {"agentic", "compare"}:
            agent_args = [PYTHON, "scripts/74_run_agentic_market_view.py", "--frequency", args.agent_frequency]
            if args.agent_config:
                agent_args.extend(["--config", args.agent_config])
            run_command(
                "generate Agentic Market Views",
                agent_args,
            )

    if args.refresh_daily_itransformer:
        run_command("build daily-path iTransformer dataset", [PYTHON, "scripts/65_build_itransformer_daily_dataset.py"])
        run_command("train daily-path iTransformer", [PYTHON, "scripts/66_train_itransformer_daily.py"])

    if args.refresh_daily_itransformer_v2:
        run_command("build iTransformer V2 timing dataset", [PYTHON, "scripts/119_build_itransformer_daily_timing_v2.py"])
        run_command("walk-forward train iTransformer V2", [PYTHON, "scripts/120_train_itransformer_daily_timing_v2.py"])
        run_command("materialize iTransformer V2 monthly OOS predictions", [PYTHON, "scripts/121_evaluate_itransformer_v2_portfolio_overlay.py", "--materialize-only"])
        run_command("audit iTransformer V2 point-in-time chain", [PYTHON, "scripts/122_audit_itransformer_v2_point_in_time.py"])

    synthetic_outputs = build_common_synthetic_assets()
    if synthetic_outputs:
        print(json.dumps({"synthetic_assets": synthetic_outputs}, ensure_ascii=False), flush=True)

    versions = ["no_agent", "agentic"] if args.strategy_version == "compare" else [args.strategy_version]
    all_results: list[dict[str, Any]] = []
    for version in versions:
        print(f"\n=== run {version} product profiles: {', '.join(profiles)} ===", flush=True)
        results: list[dict[str, Any]] = []
        if args.no_parallel or len(profiles) == 1:
            for profile in profiles:
                result = run_profile(profile, version)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                results.append(result)
        else:
            workers = max(1, min(args.workers, len(profiles)))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_profile, profile, version): profile for profile in profiles}
                for future in as_completed(futures):
                    result = future.result()
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    results.append(result)
        results = sorted(results, key=lambda x: profiles.index(x["profile"]))
        write_manifest(results, version)
        all_results.extend(results)

    run_hybrid = not args.skip_itransformer_ppo and set(profiles) == {"low_vol", "balanced", "high_vol"}
    hybrid_versions: set[str] = set()
    if run_hybrid:
        for version in versions:
            hybrid_args = [PYTHON, "scripts/123_run_itransformer_ppo_pipeline.py"]
            if version == "agentic":
                hybrid_args.extend(["--config", "config/itransformer_ppo_agentic_config.yaml"])
            run_command(f"run {version} constrained iTransformer V2 + PPO enhancement", hybrid_args)
            hybrid_versions.add(version)
    elif not args.skip_itransformer_ppo:
        print("Skip iTransformer+PPO enhancement because all three profiles are required for the joint report.", flush=True)

    if not args.skip_visuals and args.strategy_version == "no_agent":
        run_command("generate V1 visual report", [PYTHON, "scripts/51_visualize_v1_results.py", "--profiles", ",".join(profiles)])
        if "no_agent" in hybrid_versions:
            run_command("generate iTransformer+PPO visual comparison", [PYTHON, "scripts/124_visualize_itransformer_ppo.py"])
    elif not args.skip_visuals and args.strategy_version in {"agentic", "compare"}:
        run_command("generate dual-version comparison", [PYTHON, "scripts/75_compare_agentic_versions.py", "--profiles", ",".join(profiles)])
        if "no_agent" in hybrid_versions:
            run_command("generate no-agent iTransformer+PPO visual comparison", [PYTHON, "scripts/124_visualize_itransformer_ppo.py"])
        if "agentic" in hybrid_versions:
            run_command(
                "generate agentic iTransformer+PPO visual comparison",
                [PYTHON, "scripts/124_visualize_itransformer_ppo.py", "--config", "config/itransformer_ppo_agentic_config.yaml"],
            )

    print("\n=== done ===", flush=True)
    print("Manifests written under output/reports/.", flush=True)


if __name__ == "__main__":
    main()
