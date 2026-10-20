from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.market_skill import MarketSkillPaths, build_ep_views, cards_for_llm_frequency


def _read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _date_key(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _load_existing_audit(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("date"):
            out[str(item["date"])] = item
    return out


def _fallback_rows_for_date(date: str, reason: str) -> pd.DataFrame:
    mechanical_path = MarketSkillPaths().mechanical_views
    if not mechanical_path.exists():
        return pd.DataFrame()
    mechanical = pd.read_parquet(mechanical_path)
    mask = pd.to_datetime(mechanical["date"], errors="coerce").dt.strftime("%Y-%m-%d").eq(date)
    rows = mechanical[mask].copy()
    if not rows.empty:
        rows["generator"] = "agentic_fallback_mechanical_v1"
        rows["model_alignment"] = reason
    return rows


def _read_one_date_outputs(out_dir: Path, date: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    views_path = out_dir / "market_skill_views_agentic_one.parquet"
    audit_path = out_dir / "agentic_audit_one.jsonl"
    raw_path = out_dir / "agentic_raw_one.csv"

    views = pd.DataFrame()
    if views_path.exists():
        tmp = pd.read_parquet(views_path)
        if not tmp.empty and "date" in tmp.columns:
            mask = pd.to_datetime(tmp["date"], errors="coerce").dt.strftime("%Y-%m-%d").eq(date)
            views = tmp[mask].copy()

    audit: dict[str, Any] = {}
    if audit_path.exists():
        lines = [x for x in audit_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines:
            candidates = [json.loads(x) for x in lines]
            audit = next((x for x in candidates if str(x.get("date")) == date), candidates[0])

    raw = pd.DataFrame()
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        if "date" in raw.columns:
            raw = raw[raw["date"].astype(str).eq(date)].copy()
    return views, audit, raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust hidden-date Agentic Market View runner with per-date timeout.")
    parser.add_argument("--config", default="config/agentic_market_view_deepseek.yaml")
    parser.add_argument("--frequency", default="month_end", choices=["latest", "month_end", "all"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--per-date-timeout", type=int, default=480)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--tmp-dir", default="output/reports/agentic_robust_tmp")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg = _read_config(config_path)
    paths_cfg = cfg["paths"]
    views_path = ROOT / paths_cfg["agentic_views"]
    ep_path = ROOT / paths_cfg["agentic_ep_views"]
    audit_path = ROOT / paths_cfg["audit_jsonl"]
    raw_path = ROOT / paths_cfg["raw_responses"]
    report_path = ROOT / paths_cfg["report"]
    progress_path = report_path.with_name("agentic_market_view_robust_progress.log")
    tmp_root = ROOT / args.tmp_dir

    for path in [views_path, ep_path, audit_path, raw_path, report_path, progress_path, tmp_root]:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    cards = pd.read_parquet(ROOT / "data_processed" / "market_state_cards_daily_v1.parquet")
    if args.dates:
        wanted = set(args.dates)
        selected = cards[pd.to_datetime(cards["date"]).dt.strftime("%Y-%m-%d").isin(wanted)].copy()
    else:
        selected = cards_for_llm_frequency(cards, args.frequency)
    selected = selected.sort_values("date").reset_index(drop=True)
    if args.max_dates and args.max_dates > 0:
        selected = selected.head(args.max_dates).copy()

    target_dates = [_date_key(x) for x in selected["date"]]
    existing = _load_existing_audit(audit_path) if args.resume else {}
    done_dates = set(existing) if args.resume else set()
    run_dates = [d for d in target_dates if d not in done_dates]

    progress_path.write_text(
        f"Robust Agentic run started: total_dates={len(target_dates)} run_dates={len(run_dates)} "
        f"workers={args.workers} timeout={args.per_date_timeout}s hidden_calendar_date=true\n",
        encoding="utf-8",
    )

    all_rows: list[pd.DataFrame] = []
    all_audits: dict[str, dict[str, Any]] = dict(existing)
    all_raw: list[pd.DataFrame] = []
    if args.resume and views_path.exists():
        old_views = pd.read_parquet(views_path)
        if not old_views.empty:
            old_views["date_key"] = pd.to_datetime(old_views["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            all_rows.append(old_views[old_views["date_key"].isin(done_dates)].drop(columns=["date_key"]).copy())
    if args.resume and raw_path.exists():
        old_raw = pd.read_csv(raw_path)
        if not old_raw.empty and "date" in old_raw.columns:
            all_raw.append(old_raw[old_raw["date"].astype(str).isin(done_dates)].copy())

    def materialize() -> pd.DataFrame:
        completed_dates = set(all_audits)
        out = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
        mechanical = pd.read_parquet(MarketSkillPaths().mechanical_views)
        mechanical["date"] = pd.to_datetime(mechanical["date"], errors="coerce")
        untouched = mechanical[~mechanical["date"].dt.strftime("%Y-%m-%d").isin(completed_dates)].copy()
        out = pd.concat([untouched, out], ignore_index=True, sort=False)
        if not out.empty:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.sort_values(["date", "validator_status", "source_asset", "target_asset"]).reset_index(drop=True)
        out.to_parquet(views_path, index=False)
        ep = build_ep_views(out, paths=MarketSkillPaths(ep_views=ep_path))
        audit_items = [all_audits[k] for k in sorted(all_audits)]
        audit_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in audit_items), encoding="utf-8")
        raw = pd.concat(all_raw, ignore_index=True, sort=False) if all_raw else pd.DataFrame()
        raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
        fallback_count = sum(bool(x.get("fallback")) for x in audit_items)
        report_path.write_text(
            "# Agentic Market View Robust Run\n\n"
            f"- completed_dates: {len(audit_items)}\n- total_dates: {len(target_dates)}\n"
            f"- views: {len(out)}\n- ep_views: {len(ep)}\n- fallback_dates: {fallback_count}\n"
            f"- hidden_calendar_date: true\n",
            encoding="utf-8",
        )
        return out

    def run_one(date: str) -> tuple[str, pd.DataFrame, dict[str, Any], pd.DataFrame, str]:
        out_dir = tmp_root / date.replace("-", "")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "79_run_agentic_one_date.py"),
            "--date",
            date,
            "--config",
            args.config,
            "--out-dir",
            str(out_dir),
        ]
        try:
            subprocess.run(cmd, cwd=ROOT, check=True, timeout=args.per_date_timeout, capture_output=True, text=True, env=os.environ.copy())
            views, audit, raw = _read_one_date_outputs(out_dir, date)
            if audit:
                audit["runner_status"] = "ok"
            else:
                audit = {"date": date, "fallback": True, "fallback_reason": "missing_child_audit", "runner_status": "missing_child_audit"}
            return date, views, audit, raw, "ok"
        except subprocess.TimeoutExpired:
            rows = _fallback_rows_for_date(date, "per_date_timeout")
            audit = {
                "date": date,
                "decision_id": None,
                "fallback": True,
                "fallback_reason": "per_date_timeout",
                "runner_status": "timeout",
                "timeout_seconds": args.per_date_timeout,
                "card_scope": {"calendar_date_hidden": True},
                "errors": [{"stage": "runner", "error": "per_date_timeout"}],
            }
            return date, rows, audit, pd.DataFrame(), "timeout"
        except subprocess.CalledProcessError as exc:
            rows = _fallback_rows_for_date(date, "child_process_error")
            audit = {
                "date": date,
                "decision_id": None,
                "fallback": True,
                "fallback_reason": "child_process_error",
                "runner_status": "error",
                "card_scope": {"calendar_date_hidden": True},
                "errors": [{"stage": "runner", "error": str(exc), "stderr": exc.stderr[-2000:] if exc.stderr else ""}],
            }
            return date, rows, audit, pd.DataFrame(), "error"

    workers = max(1, int(args.workers))
    completed = len(done_dates)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, d): d for d in run_dates}
        for future in as_completed(futures):
            date, views, audit, raw, status = future.result()
            if not views.empty:
                all_rows.append(views)
            all_audits[date] = audit
            if not raw.empty:
                all_raw.append(raw)
            completed += 1
            out = materialize()
            line = (
                f"{completed}/{len(target_dates)} date={date} status={status} "
                f"fallback={bool(audit.get('fallback'))} reason={audit.get('fallback_reason', audit.get('runner_status', ''))} "
                f"views={len(views)} completed_dates={len(all_audits)}\n"
            )
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            print(line.strip(), flush=True)
    materialize()


if __name__ == "__main__":
    main()
