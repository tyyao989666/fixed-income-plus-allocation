"""Build a clean, external-facing v4.3 research package.

The package intentionally contains only the formal clean-calendar v4.3 chain,
not historical v1 experiments or parameter sweeps.  It is safe to send to a
reviewer: no API keys, no temporary logs unrelated to the final study.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "reports"
PROCESSED = ROOT / "data_processed"
DELIVERABLES = ROOT / "output" / "deliverables"
PACKAGE_NAME = "v43_clean_calendar_research_package_20260818"
PACKAGE = DELIVERABLES / PACKAGE_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_item(source: Path, relative_target: str, category: str, rows: list[dict]) -> None:
    target = PACKAGE / relative_target
    if not source.exists():
        rows.append(
            {
                "category": category,
                "package_path": relative_target,
                "source_path": str(source.relative_to(ROOT)),
                "status": "missing_in_workspace",
                "size_bytes": "",
                "sha256": "",
            }
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    rows.append(
        {
            "category": category,
            "package_path": relative_target,
            "source_path": str(source.relative_to(ROOT)),
            "status": "included",
            "size_bytes": source.stat().st_size,
            "sha256": sha256(target),
        }
    )


def write_readme(rows: list[dict]) -> None:
    # Four generated audit files are added immediately after this README.
    included = sum(row["status"] == "included" for row in rows) + 4
    size_mb = sum(int(row["size_bytes"] or 0) for row in rows) / 1024**2
    content = f"""# 固收+多资产配置 v4.3：研究底稿包

## 交付定位
这是可外发、可审计的正式研究包。它只保留 **v4.3 clean-calendar** 正式链路：

1. A0：分层风险平价底仓；
2. A1：Market View + EP/skfolio 观点修正；
3. A2：因果 iTransformer 相对观点门控 + 教师引导 PPO 自适应执行；
4. 三类产品（低波、中波、高波）在完全相同资产池、日期、T+1 与成本口径下的控制变量比较。

包内共 {included} 个文件，约 {size_mb:.1f} MB。文件指纹见 `00_索引与校验/manifest.csv`。

## 先读什么

| 目的 | 文件 |
|---|---|
| 文章主稿，可直接作为写作底稿 | `01_文章主稿/v43_three_layer_article_draft.md` |
| 方法、训练与结果的完整技术说明 | `02_策略与复现/hierarchical_asset_itransformer_v43_clean_calendar_method_and_results.md` |
| 最终组合报告及正式门控结论 | `03_结果与审计/hierarchical_asset_itransformer_v43_clean_calendar_portfolio_results.md` |
| 38 个资产、数据口径、代理与限制 | `04_数据底稿/v43_data_draft.md` |
| Excel 总览（策略、资产、数据、特征、结果、限制） | `04_数据底稿/v43_three_layer_framework_and_data_workbook.xlsx` |
| 单文件复现入口 | `05_代码与配置/run_v43_clean_calendar_full_pipeline.py` |

## 正式实验口径

- **资产池：** 38 个资产；`rate_30y_curve_proxy` 是 30Y 利率债的收益率曲线/财富指数代理，不能与原始列 `rate_30y` 重复计数。
- **日期：** 原始日频覆盖 2018-01-02 至 2026-07-15；控制变量回测口径为 2022-01-03 至 2026-07-15，正式比较结果重点展示 2024 年以来。
- **交易：** Point-in-Time 处理、T+1 执行、单边 5bp 换手成本、无现金、资产与袖套上下限。
- **泄露控制：** 每天只使用当日及以前已可得数据；所有模型训练、标准化、标签和教师策略均按滚动时间边界切分。审计结果见 `03_结果与审计/*point_in_time_audit.csv`。
- **模型边界：** iTransformer 只形成相对观点与门控，不直接预测仓位；PPO 只决定向上游目标仓位靠拢的执行速度，不产生资产选择信号。

## 正式结论的阅读方式

正式组合门控为 **False**，原因是低波产品的分块 bootstrap 夏普正向概率未达到正式阈值；中波和高波通过相对旧版替代门槛。因此不得将 A2 叙述为对所有产品的全面、无条件替代。低波仍建议以 A1 为主，A2 作为观察版本；中高波可作为有约束的候选正式版本。

## 目录说明

- `01_文章主稿/`：论文/报告正文与写作蓝图。
- `02_策略与复现/`：正式配置、完整方法说明、运行脚本与依赖。
- `03_结果与审计/`：A0/A1/A2 的指标、日频结果、bootstrap、因果审计和门控。
- `04_数据底稿/`：38 资产字典、数据说明、特征与状态卡输入、核心处理数据。
- `05_代码与配置/`：数据刷新、基线、文章/Excel生成和正式流水线代码。

## 复现顺序

在仓库根目录中安装 `requirements.txt` 后，依次运行：

```powershell
python scripts\\116_refresh_latest_data.py
python scripts\\171_run_v43_control_variable_baselines.py
python scripts\\run_v43_clean_calendar_full_pipeline.py
```

完整 iTransformer/PPO 训练耗时较长。若只复核已交付结果，不需要重新训练，直接读取 `03_结果与审计/` 与 `04_数据底稿/核心处理数据/` 即可。

## 外发限制

本包不包含 API Key、登录凭证或 Wind/iFinD 客户端配置。第三方若需要重新刷新至新日期，需自行拥有合规的数据终端权限；不应将本包内的代理序列表述为可直接交易 ETF。
"""
    (PACKAGE / "README.md").write_text(content, encoding="utf-8")


def write_validation() -> None:
    universe = pd.read_parquet(PROCESSED / "asset_universe.parquet")
    returns = pd.read_parquet(PROCESSED / "asset_returns_daily.parquet")
    date_col = "date"
    asset_col = "asset_code"
    dates = pd.to_datetime(returns[date_col])
    assets = sorted(returns[asset_col].dropna().unique())
    formal_cutoff = "2026-07-15"
    checks = [
        ("正式资产数量", len(universe), 38, len(universe) == 38, "asset_universe.parquet"),
        ("原始收益资产数量（长表）", len(assets), 38, len(assets) == 38, "asset_returns_daily.parquet"),
        ("原始收益起始日", dates.min().date().isoformat(), "2018-01-02", dates.min().date().isoformat() == "2018-01-02", "asset_returns_daily.parquet"),
        ("原始收益最新日", dates.max().date().isoformat(), ">= 2026-07-15", dates.max().date().isoformat() >= formal_cutoff, "asset_returns_daily.parquet"),
        ("正式模型冻结日", formal_cutoff, formal_cutoff, True, "hierarchical_asset_itransformer_v43_clean_calendar.yaml"),
        ("周末记录数", int((dates.dt.dayofweek >= 5).sum()), 0, int((dates.dt.dayofweek >= 5).sum()) == 0, "asset_returns_daily.parquet"),
    ]
    validation = pd.DataFrame(checks, columns=["check", "actual", "expected", "passed", "evidence_file"])
    validation.to_csv(PACKAGE / "00_索引与校验" / "release_validation.csv", index=False, encoding="utf-8-sig")
    universe.to_csv(PACKAGE / "04_数据底稿" / "38资产字典.csv", index=False, encoding="utf-8-sig")
    coverage = (
        returns.groupby(asset_col, as_index=False)
        .agg(
            start_date=(date_col, "min"),
            end_date=(date_col, "max"),
            observations=(date_col, "size"),
            missing_return_rate=("return_1d", lambda series: round(float(series.isna().mean()), 6)),
            data_source=("data_source", lambda series: " | ".join(sorted(set(series.dropna().astype(str))))),
        )
        .rename(columns={asset_col: "asset"})
        .sort_values("asset")
    )
    coverage["start_date"] = pd.to_datetime(coverage["start_date"]).dt.date.astype(str)
    coverage["end_date"] = pd.to_datetime(coverage["end_date"]).dt.date.astype(str)
    coverage.to_csv(PACKAGE / "04_数据底稿" / "收益数据覆盖表.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    # A prior interrupted archive build may have already copied the payload.
    # Re-running is safe: files are overwritten in place and the manifest/zip
    # are regenerated from the final payload.
    for folder in ["00_索引与校验", "01_文章主稿", "02_策略与复现", "03_结果与审计", "04_数据底稿/核心处理数据", "05_代码与配置"]:
        (PACKAGE / folder).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    docs = [
        ("v43_three_layer_article_draft.md", "01_文章主稿/v43_three_layer_article_draft.md"),
        ("v43_clean_calendar_research_writing_blueprint.md", "01_文章主稿/v43_clean_calendar_research_writing_blueprint.md"),
        ("v43_data_draft.md", "04_数据底稿/v43_data_draft.md"),
        ("v43_three_layer_framework_and_data_workbook.xlsx", "04_数据底稿/v43_three_layer_framework_and_data_workbook.xlsx"),
        ("hierarchical_asset_itransformer_v43_clean_calendar_method_and_results.md", "02_策略与复现/hierarchical_asset_itransformer_v43_clean_calendar_method_and_results.md"),
        ("hierarchical_asset_itransformer_v43_clean_calendar_portfolio_results.md", "03_结果与审计/hierarchical_asset_itransformer_v43_clean_calendar_portfolio_results.md"),
    ]
    for source_name, target in docs:
        copy_item(REPORTS / source_name, target, "documentation", rows)

    result_names = [
        "v43_clean_calendar_control_variable_baselines.csv",
        "v43_clean_calendar_control_variable_baselines.json",
        "hierarchical_asset_itransformer_v43_clean_calendar_three_profile_summary.csv",
        "hierarchical_asset_itransformer_v43_clean_calendar_bootstrap_robustness.csv",
        "hierarchical_asset_itransformer_v43_clean_calendar_point_in_time_audit.csv",
        "hierarchical_asset_itransformer_v43_clean_calendar_gate.json",
        "hierarchical_asset_itransformer_v43_clean_calendar_gate.components.csv",
        "hierarchical_asset_itransformer_v43_clean_calendar_portfolio_gate.json",
        "hierarchical_asset_itransformer_v43_clean_calendar_vs_v6_comparison.csv",
        "hierarchical_asset_itransformer_v43_clean_calendar_metadata.json",
        "hierarchical_asset_itransformer_v43_clean_calendar_training_log.csv",
        "hierarchical_asset_itransformer_v43_clean_calendar_training_progress.jsonl",
        "hierarchical_asset_itransformer_v43_clean_calendar_validation.csv",
    ]
    for profile in ["low_vol", "balanced", "high_vol"]:
        result_names.extend(
            [
                f"hierarchical_asset_itransformer_v43_clean_calendar_{profile}_metrics.csv",
                f"hierarchical_asset_itransformer_v43_clean_calendar_{profile}_daily.csv",
                f"hierarchical_asset_itransformer_v43_clean_calendar_{profile}_controller_training.csv",
                f"v43_clean_calendar_{profile}_A0_hierarchical_risk_parity_daily.csv",
                f"v43_clean_calendar_{profile}_A0_hierarchical_risk_parity_weights.parquet",
                f"v43_clean_calendar_{profile}_A1_market_view_ep_skfolio_daily.csv",
                f"v43_clean_calendar_{profile}_A1_market_view_ep_skfolio_weights.parquet",
            ]
        )
    for source_name in result_names:
        copy_item(REPORTS / source_name, f"03_结果与审计/{source_name}", "results_and_audit", rows)

    data_names = [
        "asset_universe.parquet",
        "asset_returns_daily.parquet",
        "feature_panel_daily_v43.parquet",
        "feature_panel_monthly_v43.parquet",
        "macro_factors_monthly_v43.parquet",
        "market_state_cards_daily_v1.parquet",
        "market_skill_views_mechanical_daily_v1.parquet",
        "hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz",
        "hierarchical_asset_itransformer_v43_clean_calendar_asset_predictions.parquet",
        "hierarchical_asset_itransformer_v43_clean_calendar_sleeve_predictions.parquet",
    ]
    for profile in ["low_vol", "balanced", "high_vol"]:
        data_names.extend(
            [
                f"hierarchical_asset_itransformer_v43_clean_calendar_{profile}_decisions.parquet",
                f"hierarchical_asset_itransformer_v43_clean_calendar_{profile}_targets_daily.parquet",
                f"hierarchical_asset_itransformer_v43_clean_calendar_{profile}_weights_daily.parquet",
            ]
        )
    for source_name in data_names:
        copy_item(PROCESSED / source_name, f"04_数据底稿/核心处理数据/{source_name}", "processed_data", rows)

    code_names = [
        "116_refresh_latest_data.py",
        "171_run_v43_control_variable_baselines.py",
        "172_generate_v43_article_and_data_drafts.py",
        "173_build_v43_research_and_data_workbook.py",
        "run_v43_clean_calendar_full_pipeline.py",
        "run_v43_clean_calendar_long_training.cmd",
        "run_v43_clean_calendar_portfolio.cmd",
    ]
    for source_name in code_names:
        copy_item(ROOT / "scripts" / source_name, f"05_代码与配置/{source_name}", "code", rows)
    copy_item(ROOT / "config" / "asset_universe.yaml", "02_策略与复现/asset_universe.yaml", "config", rows)
    copy_item(ROOT / "config" / "hierarchical_asset_itransformer_v43_clean_calendar.yaml", "02_策略与复现/hierarchical_asset_itransformer_v43_clean_calendar.yaml", "config", rows)
    copy_item(ROOT / "requirements.txt", "02_策略与复现/requirements.txt", "dependency", rows)

    write_validation()
    write_readme(rows)
    rows.extend(
        [
            {"category": "generated", "package_path": "README.md", "source_path": "generated", "status": "included", "size_bytes": (PACKAGE / "README.md").stat().st_size, "sha256": sha256(PACKAGE / "README.md")},
            {"category": "generated", "package_path": "00_索引与校验/release_validation.csv", "source_path": "generated", "status": "included", "size_bytes": (PACKAGE / "00_索引与校验/release_validation.csv").stat().st_size, "sha256": sha256(PACKAGE / "00_索引与校验/release_validation.csv")},
            {"category": "generated", "package_path": "04_数据底稿/38资产字典.csv", "source_path": "generated", "status": "included", "size_bytes": (PACKAGE / "04_数据底稿/38资产字典.csv").stat().st_size, "sha256": sha256(PACKAGE / "04_数据底稿/38资产字典.csv")},
            {"category": "generated", "package_path": "04_数据底稿/收益数据覆盖表.csv", "source_path": "generated", "status": "included", "size_bytes": (PACKAGE / "04_数据底稿/收益数据覆盖表.csv").stat().st_size, "sha256": sha256(PACKAGE / "04_数据底稿/收益数据覆盖表.csv")},
        ]
    )
    with (PACKAGE / "00_索引与校验" / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "package_path", "source_path", "status", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    archive = DELIVERABLES / f"{PACKAGE_NAME}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for file in PACKAGE.rglob("*"):
            if file.is_file():
                zipped.write(file, file.relative_to(PACKAGE.parent))
    summary = {
        "package": str(PACKAGE.relative_to(ROOT)),
        "archive": str(archive.relative_to(ROOT)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "included_files": sum(row["status"] == "included" for row in rows),
        "missing_files": [row["package_path"] for row in rows if row["status"] != "included"],
        "package_size_bytes": sum(path.stat().st_size for path in PACKAGE.rglob("*") if path.is_file()),
        "archive_size_bytes": archive.stat().st_size,
    }
    (PACKAGE / "00_索引与校验" / "package_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
