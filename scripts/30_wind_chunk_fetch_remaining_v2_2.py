from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SKILL_CLI = Path(r"C:\Users\lenovo\.agents\skills\wind-mcp-skill\scripts\cli.mjs")


def wind_analytics(question: str, cache: Path):
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8")), "ok_cached"
    params = {"question": question.replace(" ", ""), "lang": "中文"}
    cmd = [
        "node",
        str(SKILL_CLI),
        "call",
        "analytics_data",
        "get_financial_data",
        json.dumps(params, ensure_ascii=False, separators=(",", ":")),
    ]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", timeout=180)
    if res.returncode != 0:
        return {"error": res.stdout.strip() or res.stderr.strip()}, "failed"
    try:
        outer = json.loads(res.stdout)
        payload = json.loads(outer["content"][0]["text"])
        if payload.get("error"):
            return {"error": payload["error"]}, "failed"
        data = payload.get("data")
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, "ok"
    except Exception as exc:
        return {"error": f"{exc}: {res.stdout[:300]}"}, "failed_parse"


def tables_to_frames(payload, dataset: str, year: int) -> list[pd.DataFrame]:
    frames = []
    blocks = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(blocks, list):
        return frames
    for i, block in enumerate(blocks):
        if not isinstance(block, dict) or "columns" not in block or "rows" not in block:
            continue
        cols = [str(c.get("name", f"col_{j}")) for j, c in enumerate(block.get("columns", []))]
        rows = block.get("rows", [])
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
        df.insert(0, "source_year", year)
        df.insert(0, "table_no", i)
        df.insert(0, "dataset", dataset)
        df["resolved_question"] = block.get("resolved_question", "")
        frames.append(df)
    return frames


def main() -> None:
    raw = ROOT / "data_raw" / "wind"
    processed = ROOT / "data_processed"
    reports = ROOT / "output" / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    templates = {
        "cb_balance_supply": "查询{year}年1月至{year}年12月可转债市场余额、存量只数、新券发行数量、退市数量、强赎数量月频数据,返回年份、月份、可转债数量、可转债余额、新发行数量、退市数量、强赎数量。",
        "cb_force_redeem_amount": "查询{year}年1月至{year}年12月可转债强制赎回规模、强赎退市规模月频数据,返回年份、月份、强赎规模。",
        "credit_default": "查询{year}年1月至{year}年12月信用债违约规模、违约只数月频数据,返回年份、月份、违约规模、违约只数。",
        "credit_cancel": "查询{year}年1月至{year}年12月债券取消发行规模、取消发行只数月频数据,返回年份、月份、取消发行规模、取消发行只数。",
        "rating_downgrade": "查询{year}年1月至{year}年12月债券主体评级下调数量、负面评级行动数量月频数据,返回年份、月份、评级下调数量、负面评级行动数量。",
    }
    years = range(2018, 2027)
    all_frames = []
    rows = []

    for dataset, template in templates.items():
        for year in years:
            end_month = 6 if year == 2026 else 12
            q = template.format(year=year).replace(f"{year}年12月", f"{year}年{end_month}月")
            cache = raw / f"analytics_chunk_{dataset}_{year}.json"
            data, status = wind_analytics(q, cache)
            frames = tables_to_frames(data, dataset, year) if status.startswith("ok") else []
            row_count = sum(len(f) for f in frames)
            rows.append(
                {
                    "dataset": dataset,
                    "year": year,
                    "status": status,
                    "rows": row_count,
                    "file": str(cache.relative_to(ROOT)) if cache.exists() else "",
                    "error": "" if status.startswith("ok") else json.dumps(data, ensure_ascii=False)[:500],
                }
            )
            all_frames.extend(frames)
            time.sleep(0.2)

    report = pd.DataFrame(rows)
    report.to_excel(reports / "v2_2_wind_chunk_fetch_remaining_report.xlsx", index=False)
    report.to_csv(reports / "v2_2_wind_chunk_fetch_remaining_report.csv", index=False, encoding="utf-8-sig")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True, sort=False)
        combined.to_parquet(processed / "wind_remaining_state_engine_chunk_raw.parquet", index=False)
        combined.to_csv(processed / "wind_remaining_state_engine_chunk_raw.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# v2.2 Wind 剩余缺口分年拉取报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- 成功/缓存成功任务数：{report['status'].astype(str).str.startswith('ok').sum()}",
        f"- 失败任务数：{(~report['status'].astype(str).str.startswith('ok')).sum()}",
        f"- 合并表行数：{sum(len(f) for f in all_frames)}",
        "",
        "## 按数据集统计",
    ]
    if not report.empty:
        for dataset, g in report.groupby("dataset"):
            lines.append(f"- {dataset}: 成功{g['status'].astype(str).str.startswith('ok').sum()}个年份, 行数{int(g['rows'].sum())}")
    (reports / "v2_2_wind_chunk_fetch_remaining_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(report.to_string(index=False))
    print(f"\nSaved: {reports / 'v2_2_wind_chunk_fetch_remaining_report.xlsx'}")
    if all_frames:
        print(f"Saved: {processed / 'wind_remaining_state_engine_chunk_raw.parquet'}")


if __name__ == "__main__":
    main()
