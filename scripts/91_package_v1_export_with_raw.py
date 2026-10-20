from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXPORT_NAME = f"fixed_income_plus_allocation_v1_export_{date.today():%Y%m%d}"
STAGE = DIST / EXPORT_NAME
ZIP_BASE = DIST / EXPORT_NAME


RUN_ME = """# 固收+多资产动态配置 V1 外发运行说明

## 1. 环境准备

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

如果只复现已经生成好的本地数据和结果，不需要配置 API Key。

如果要重新拉取外部数据或重新生成 Agentic LLM Market View，请复制 `.env.example` 为 `.env`，填入相应密钥：

```text
BIGMODEL_API_KEY=...
DEEPSEEK_API_KEY=...
TUSHARE_TOKEN=...
WIND_API_KEY=...
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

## 5. 数据口径

- `data_raw/` 保留原始数据库和抓取落地文件，便于追溯。
- `data_processed/` 保留当前策略直接运行所需的处理后数据。
- 30Y 利率债使用收益率曲线代理构造，不是 ETF 实盘价格。
- Agentic 版本只能输出 market view，不直接输出仓位。
- 每个调仓截面只使用当期可见状态卡，不把未来收益喂给 LLM。
"""


def copy_data_raw() -> int:
    src_root = ROOT / "data_raw"
    dst_root = STAGE / "data_raw"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    copied = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        if "__pycache__" in src.parts:
            continue
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def write_clean_docs(raw_files: int) -> None:
    (STAGE / "RUN_ME.md").write_text(RUN_ME, encoding="utf-8")

    manifest_path = STAGE / "CLEAN_FILE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_database_included"] = True
    manifest["raw_database_files"] = raw_files
    manifest["excluded_by_design"] = [
        x for x in manifest.get("excluded_by_design", []) if x != "data_raw"
    ]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Clean Export Manifest",
        "",
        f"- Export name: `{EXPORT_NAME}`",
        f"- Source: `{ROOT}`",
        f"- Copied files before raw database: `{manifest.get('copied_files')}`",
        f"- Raw database files included: `{raw_files}`",
        "",
        "## Included",
        "",
        "- runnable source code under `src/`",
        "- selected pipeline scripts under `scripts/`",
        "- all strategy/config YAML/JSON files under `config/`",
        "- original source database/files under `data_raw/`",
        "- current core processed data under `data_processed/`",
        "- current no-agent and agentic result files under `output/reports/`",
        "- reproduction manuals and visual report",
        "",
        "## Excluded",
        "",
        "- old grid-search and tuning debris",
        "- cache folders and `__pycache__`",
        "- temporary logs and pid files",
        "- real API keys",
    ]
    missing = manifest.get("missing_optional_files", [])
    if missing:
        lines += ["", "## Missing Optional Files", ""]
        lines += [f"- `{x}`" for x in missing]
    (STAGE / "CLEAN_FILE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    subprocess.run([sys.executable, "scripts/90_package_v1_export.py"], cwd=ROOT, check=True)
    raw_files = copy_data_raw()
    write_clean_docs(raw_files)
    archive = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=DIST, base_dir=EXPORT_NAME)
    total_files = sum(1 for x in STAGE.rglob("*") if x.is_file())
    total_bytes = sum(x.stat().st_size for x in STAGE.rglob("*") if x.is_file())
    print(json.dumps({
        "stage": str(STAGE),
        "zip": archive,
        "total_files": total_files,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "raw_database_files": raw_files,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
