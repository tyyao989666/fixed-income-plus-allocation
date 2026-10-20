import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agentic_market_view import (  # noqa: E402
    _agent_rows,
    _single_card_scope,
    build_graph,
    load_agent_config,
)
from src.market_skill import (  # noqa: E402
    MarketSkillPaths,
    _compact_card_for_prompt,
    build_ep_views,
    load_config as load_market_skill_config,
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in sorted(rows, key=lambda x: x["date"])),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun failed/fallback Agentic Market View dates and merge outputs.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "agentic_market_view_deepseek.yaml")
    parser.add_argument("--dates", default=None, help="Comma-separated YYYY-MM-DD dates. Defaults to fallback dates from audit.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    agent_cfg = load_agent_config(args.config)
    market_cfg = load_market_skill_config()
    paths_cfg = agent_cfg["paths"]
    views_path = ROOT / paths_cfg["agentic_views"]
    ep_path = ROOT / paths_cfg["agentic_ep_views"]
    audit_path = ROOT / paths_cfg["audit_jsonl"]
    raw_path = ROOT / paths_cfg["raw_responses"]
    report_path = ROOT / paths_cfg["report"]
    progress_path = report_path.with_name("agentic_market_view_rerun_progress.log")

    audits_old = load_jsonl(audit_path)
    if args.dates:
        target_dates = sorted({x.strip() for x in args.dates.split(",") if x.strip()})
    else:
        target_dates = sorted({x["date"] for x in audits_old if bool(x.get("fallback"))})
    if not target_dates:
        print("No target dates to rerun.")
        return

    cards = pd.read_parquet(ROOT / "data_processed" / "market_state_cards_daily_v1.parquet")
    cards["date_key"] = pd.to_datetime(cards["date"]).dt.strftime("%Y-%m-%d")
    selected = cards[cards["date_key"].isin(target_dates)].drop(columns=["date_key"]).copy()
    missing = sorted(set(target_dates) - set(pd.to_datetime(selected["date"]).dt.strftime("%Y-%m-%d")))
    if missing:
        raise RuntimeError(f"Missing target dates in cards: {missing}")

    graph = build_graph(agent_cfg, market_cfg)
    progress_path.write_text(f"Rerun started: total_dates={len(selected)}\n", encoding="utf-8")

    new_rows: list[dict] = []
    new_audits: list[dict] = []
    new_raw: list[dict] = []

    def execute(card_dict: dict) -> tuple[pd.Series, dict]:
        card = pd.Series(card_dict)
        payload = _compact_card_for_prompt(card)
        card_scope = _single_card_scope(payload)
        result = graph.invoke({"card": payload, "findings": [], "errors": [], "raw_responses": []})
        result["card_scope"] = card_scope
        return card, result

    workers = min(max(1, args.workers), len(selected))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, row.to_dict()) for _, row in selected.iterrows()]
        for idx, future in enumerate(as_completed(futures), start=1):
            card, result = future.result()
            rows, fallback = _agent_rows(card, result, market_cfg, agent_cfg)
            date = str(pd.Timestamp(card["date"]).date())
            errors = result.get("errors", [])
            new_rows.extend(rows)
            new_audits.append(
                {
                    "date": date,
                    "fallback": fallback,
                    "card_scope": result.get("card_scope", {}),
                    "plan": result.get("plan", {}),
                    "findings": result.get("findings", []),
                    "reviewed": result.get("reviewed", []),
                    "supervisor": result.get("supervisor", {}),
                    "errors": errors,
                }
            )
            for raw in result.get("raw_responses", []):
                new_raw.append({"date": date, **raw})
            line = f"{idx}/{len(selected)} date={date} status={'fallback' if fallback else 'agentic'} views={len(rows)} errors={len(errors)}\n"
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            print(line.strip(), flush=True)

    target_set = set(target_dates)
    audits_merged = [row for row in audits_old if row["date"] not in target_set] + new_audits
    write_jsonl(audit_path, audits_merged)

    raw_old = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    raw_new = pd.DataFrame(new_raw)
    if not raw_old.empty and "date" in raw_old.columns:
        raw_old = raw_old[~pd.to_datetime(raw_old["date"]).dt.strftime("%Y-%m-%d").isin(target_set)].copy()
    raw_merged = pd.concat([raw_old, raw_new], ignore_index=True, sort=False)
    raw_merged.to_csv(raw_path, index=False, encoding="utf-8-sig")

    views_old = pd.read_parquet(views_path) if views_path.exists() else pd.DataFrame()
    if not views_old.empty and "date" in views_old.columns:
        views_old = views_old[~pd.to_datetime(views_old["date"]).dt.strftime("%Y-%m-%d").isin(target_set)].copy()
    views_new = pd.DataFrame(new_rows)
    if not views_new.empty:
        views_new["date"] = pd.to_datetime(views_new["date"], errors="coerce")
    views_merged = pd.concat([views_old, views_new], ignore_index=True, sort=False)

    if not views_merged.empty:
        views_merged["date"] = pd.to_datetime(views_merged["date"], errors="coerce")
        views_merged = views_merged.sort_values(["date", "validator_status", "source_asset", "target_asset"]).reset_index(drop=True)
    views_merged.to_parquet(views_path, index=False)
    ep = build_ep_views(views_merged, paths=MarketSkillPaths(ep_views=ep_path))

    fallback_count = sum(bool(x.get("fallback")) for x in audits_merged)
    valid = int(views_merged["validator_status"].eq("valid").sum()) if not views_merged.empty else 0
    report_path.write_text(
        "# Agentic Market View Run\n\n"
        f"- completed_dates: {len(audits_merged)}\n- total_dates: {len(audits_merged)}\n"
        f"- rerun_dates: {len(target_dates)}\n- views: {len(views_merged)}\n- valid_views: {valid}\n"
        f"- fallback_dates: {fallback_count}\n- ep_views: {len(ep)}\n",
        encoding="utf-8",
    )
    print(f"Rerun merged: dates={len(target_dates)}, fallback_after={fallback_count}, views={len(views_merged)}, ep={len(ep)}")


if __name__ == "__main__":
    main()
