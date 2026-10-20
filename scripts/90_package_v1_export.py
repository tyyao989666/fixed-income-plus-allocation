from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXPORT_NAME = f"fixed_income_plus_allocation_v1_export_{date.today():%Y%m%d}"
STAGE = DIST / EXPORT_NAME
ZIP_BASE = DIST / EXPORT_NAME


ROOT_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
]

SCRIPTS = [
    "run_v1_full_pipeline.py",
    "10_generate_market_state_cards.py",
    "11_run_llm_scenario_engine.py",
    "12_build_ep_views.py",
    "47_build_v43_basic_data.py",
    "48_audit_v43_no_annual_data.py",
    "51_visualize_v1_results.py",
    "52_make_v1_data_usage_workbook.py",
    "63_fetch_wind_nanhua_alt_indices.py",
    "64_build_wind_alt_assets.py",
    "65_build_itransformer_daily_dataset.py",
    "66_train_itransformer_daily.py",
    "67_compare_v1_static_benchmarks.py",
    "69_fetch_aggressive_equity_assets.py",
    "70_run_itransformer_daily_enhanced.py",
    "71_blend_itransformer_daily_signals.py",
    "72_compare_blended_itransformer_portfolios.py",
    "73_walkforward_itransformer_daily.py",
    "74_run_agentic_market_view.py",
    "75_compare_agentic_versions.py",
    "76_validate_agentic_pipeline.py",
    "78_run_agentic_market_view_robust.py",
    "79_run_agentic_one_date.py",
    "80_calibrate_agentic_views.py",
    "82_run_agentic_market_view_compressed.py",
    "84_materialize_agentic_month_replacement.py",
    "85_build_agentic_profile_view_sets.py",
    "90_package_v1_export.py",
]

CORE_DATA = [
    "asset_returns_daily.parquet",
    "asset_returns_monthly.parquet",
    "asset_universe.parquet",
    "base_asset_prices_daily.parquet",
    "feature_panel_daily_v43.parquet",
    "feature_panel_monthly_v43.parquet",
    "market_state_cards_daily_v1.parquet",
    "market_skill_views_mechanical_daily_v1.parquet",
    "market_skill_views_daily_v1.parquet",
    "market_skill_views_llm_daily_v1.parquet",
    "market_skill_views_agentic_daily_v1.parquet",
    "market_skill_views_agentic_low_vol_daily_v1.parquet",
    "market_skill_views_agentic_blended_daily_v1.parquet",
    "ep_views_daily_v1.parquet",
    "ep_views_agentic_daily_v1.parquet",
    "ep_views_agentic_low_vol_daily_v1.parquet",
    "ep_views_agentic_balanced_daily_v1.parquet",
    "ep_views_agentic_high_vol_daily_v1.parquet",
    "itransformer_predictions_daily_path_monthly.parquet",
    "asset_itransformer_predictions_monthly.parquet",
    "rate_curves_daily_v43.parquet",
    "rate_30y_curve_proxy_monthly.parquet",
    "credit_spreads_daily_v43.parquet",
    "credit_spreads_strict_daily_v42.parquet",
    "credit_turnover_daily.parquet",
    "credit_turnover_by_type_daily.parquet",
    "credit_tail_risk_monthly_v42.parquet",
    "funding_liquidity_daily_v43.parquet",
    "macro_factors_monthly_v43.parquet",
    "industrial_profit_monthly_v43.parquet",
    "cb_features_daily_v43.parquet",
    "equity_features_daily_v43.parquet",
    "alt_features_daily_v43.parquet",
    "wind_nanhua_alt_indices_daily.parquet",
    "wind_nanhua_alt_indices_monthly.parquet",
    "reits_distribution_yield_v43.parquet",
    "reits_distribution_yield_single_product_v43.parquet",
    "raw_wind_gold_au9999_daily.parquet",
    "raw_wind_us_10y_yield_daily.parquet",
    "raw_wind_usd_index_daily.parquet",
]

PROFILE_DATA_TEMPLATES = [
    "v1_{profile}_risk_budget_base_monthly.parquet",
    "v1_{profile}_sleeve_weights_monthly.parquet",
    "v1_{profile}_target_weights_monthly.parquet",
    "v1_{profile}_ep_posterior_monthly.parquet",
    "v1_agentic_{profile}_risk_budget_base_monthly.parquet",
    "v1_agentic_{profile}_sleeve_weights_monthly.parquet",
    "v1_agentic_{profile}_target_weights_monthly.parquet",
    "v1_agentic_{profile}_ep_posterior_monthly.parquet",
]

REPORTS = [
    "v1_visual_report.html",
    "v1_visual_manifest.json",
    "v1_no_agent_reproduction_manual.md",
    "v1_agentic_reproduction_manual.md",
    "v1_agentic_asset_level_diagnostics.md",
    "v1_agentic_effect_diagnostics.md",
    "v1_agentic_effect_diagnostics.csv",
    "v1_agentic_comparison.md",
    "v1_agentic_comparison.csv",
    "v1_static_benchmark_comparison.md",
    "v1_static_benchmark_comparison.csv",
    "v1_static_benchmark_nav.csv",
    "v1_data_usage_readme.xlsx",
    "v1_full_pipeline_manifest.json",
    "v1_agentic_full_pipeline_manifest.json",
    "agentic_profile_view_sets_report.json",
    "agentic_pipeline_validation.json",
    "agentic_market_view_report.md",
    "agentic_market_view_audit.jsonl",
    "agentic_market_view_raw_responses.csv",
    "agentic_market_view_promptbook.md",
]

PROFILE_REPORT_TEMPLATES = [
    "v1_{profile}_backtest_metrics.csv",
    "v1_{profile}_backtest_nav.csv",
    "v1_{profile}_attribution_by_sleeve.csv",
    "v1_{profile}_monthly_attribution.csv",
    "v1_{profile}_robustness_summary.csv",
    "v1_{profile}_data_quality.csv",
    "v1_{profile}_ep_signals.csv",
    "v1_{profile}_optimizer_diagnostics.csv",
    "v1_agentic_{profile}_backtest_metrics.csv",
    "v1_agentic_{profile}_backtest_nav.csv",
    "v1_agentic_{profile}_attribution_by_sleeve.csv",
    "v1_agentic_{profile}_robustness_summary.csv",
    "v1_agentic_{profile}_data_quality.csv",
    "v1_agentic_{profile}_ep_signals.csv",
    "v1_agentic_{profile}_optimizer_diagnostics.csv",
]

DOCS = [
    "itransformer_signal_design.md",
    "market_skill_design.md",
    "v1_runbook.md",
]


def copy_file(rel: str, copied: list[str], missing: list[str]) -> None:
    src = ROOT / rel
    dst = STAGE / rel
    if not src.exists():
        missing.append(rel)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel)


def copy_tree_py(rel: str, copied: list[str]) -> None:
    base = ROOT / rel
    for src in base.rglob("*.py"):
        if "__pycache__" in src.parts:
            continue
        dst = STAGE / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(src.relative_to(ROOT)).replace("\\", "/"))


def write_run_me() -> None:
    text = """# 固收+多资产动态配置 V1 外发运行说明

## 1. 环境准备

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

如果只复现已经生成好的本地数据和结果，不需要配置 API Key。

如果要重新跑 Agentic LLM Market View，请复制 `.env.example` 为 `.env`，填入：

```text
BIGMODEL_API_KEY=...
DEEPSEEK_API_KEY=...
```

## 2. 直接复现当前无 Agent + Agentic 双版本

```powershell
python scripts\\85_build_agentic_profile_view_sets.py
python scripts\\run_v1_full_pipeline.py --skip-data --skip-market-skill --strategy-version compare --profiles low_vol,balanced,high_vol --workers 3
```

输出位置：

- `output/reports/v1_*_backtest_metrics.csv`
- `output/reports/v1_agentic_*_backtest_metrics.csv`
- `output/reports/v1_agentic_comparison.csv`
- `output/reports/v1_visual_report.html`
- `data_processed/v1_*_target_weights_monthly.parquet`
- `data_processed/v1_agentic_*_target_weights_monthly.parquet`

## 3. 只跑无 Agent 版本

```powershell
python scripts\\run_v1_full_pipeline.py --skip-data --skip-market-skill --strategy-version no_agent --profiles low_vol,balanced,high_vol --workers 3
```

## 4. 重新生成 Agentic Market View

这一步会调用外部大模型，耗时和费用取决于网络与模型响应。

```powershell
python scripts\\78_run_agentic_market_view_robust.py --config config\\agentic_market_view_deepseek.yaml --frequency month_end --workers 2 --max-retries 2
python scripts\\80_calibrate_agentic_views.py
python scripts\\85_build_agentic_profile_view_sets.py
python scripts\\run_v1_full_pipeline.py --skip-data --skip-market-skill --strategy-version compare --profiles low_vol,balanced,high_vol --workers 3
```

## 5. 重要口径

- 当前包使用已落地的本地数据，不重新拉 Wind/Tushare。
- 30Y 利率债使用收益率曲线代理构造，不是 ETF 实盘价格。
- Agentic 版本只能输出 market view，不直接输出仓位。
- 每个调仓截面只使用当期可见状态卡，不把未来收益喂给 LLM。
"""
    path = STAGE / "RUN_ME.md"
    path.write_text(text, encoding="utf-8")


def write_manifest(copied: list[str], missing: list[str]) -> None:
    summary = {
        "export_name": EXPORT_NAME,
        "created_from": str(ROOT),
        "copied_files": len(copied),
        "missing_optional_files": missing,
        "excluded_by_design": [
            "__pycache__",
            ".vscode",
            "_tmp_weight_research",
            "data_raw",
            "old grid/search/ablation/intermediate outputs",
            "model checkpoints not required for direct replay",
            "logs/pid files",
            "real API keys",
        ],
        "recommended_replay_command": (
            "python scripts\\run_v1_full_pipeline.py --skip-data --skip-market-skill "
            "--strategy-version compare --profiles low_vol,balanced,high_vol --workers 3"
        ),
    }
    (STAGE / "CLEAN_FILE_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Clean Export Manifest",
        "",
        f"- Export name: `{EXPORT_NAME}`",
        f"- Source: `{ROOT}`",
        f"- Copied files: `{len(copied)}`",
        "",
        "## Included",
        "",
        "- runnable source code under `src/`",
        "- selected pipeline scripts under `scripts/`",
        "- all strategy/config YAML/JSON files under `config/`",
        "- current core processed data under `data_processed/`",
        "- current no-agent and agentic result files under `output/reports/`",
        "- reproduction manuals and visual report",
        "",
        "## Excluded",
        "",
        "- old grid-search and tuning debris",
        "- cache folders and `__pycache__`",
        "- raw data dump folder",
        "- temporary logs and pid files",
        "- real API keys",
    ]
    if missing:
        lines += ["", "## Missing Optional Files", ""]
        lines += [f"- `{x}`" for x in missing]
    (STAGE / "CLEAN_FILE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DIST.mkdir(exist_ok=True)
    if STAGE.exists():
        resolved = STAGE.resolve()
        if DIST.resolve() not in resolved.parents:
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    copied: list[str] = []
    missing: list[str] = []

    for rel in ROOT_FILES:
        copy_file(rel, copied, missing)
    copy_tree_py("src", copied)
    for script in SCRIPTS:
        copy_file(f"scripts/{script}", copied, missing)

    for src in (ROOT / "config").glob("*"):
        if src.is_file() and src.suffix.lower() in {".yaml", ".json"}:
            rel = str(src.relative_to(ROOT)).replace("\\", "/")
            copy_file(rel, copied, missing)

    for doc in DOCS:
        copy_file(f"docs/{doc}", copied, missing)

    for item in CORE_DATA:
        copy_file(f"data_processed/{item}", copied, missing)
    for profile in ["low_vol", "balanced", "high_vol"]:
        for template in PROFILE_DATA_TEMPLATES:
            copy_file(f"data_processed/{template.format(profile=profile)}", copied, missing)
        for template in PROFILE_REPORT_TEMPLATES:
            copy_file(f"output/reports/{template.format(profile=profile)}", copied, missing)
    for report in REPORTS:
        copy_file(f"output/reports/{report}", copied, missing)

    write_run_me()
    write_manifest(copied, missing)

    archive = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=DIST, base_dir=EXPORT_NAME)
    print(json.dumps({
        "stage": str(STAGE),
        "zip": archive,
        "copied_files": len(copied),
        "missing_optional_files": len(missing),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
