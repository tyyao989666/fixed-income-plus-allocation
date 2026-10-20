from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import json

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.market_skill import MarketSkillPaths, _json_loads, build_ep_views, generate_executable_mechanical_views, load_config, validate_view


def _date_key(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _mechanical_by_date(cards: pd.DataFrame, market_cfg: dict[str, Any]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    out: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for _, card in cards.iterrows():
        filters = _json_loads(card["view_filters"])
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for view in generate_executable_mechanical_views(card, market_cfg):
            status, _ = validate_view(view, filters)
            if status == "valid":
                rows[(str(view["source_asset"]), str(view["target_asset"]))] = view
        out[_date_key(card["date"])] = rows
    return out


def calibrate() -> pd.DataFrame:
    cfg_path = ROOT / "config" / "agentic_market_view_deepseek.yaml"
    agent_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    enh = agent_cfg.get("agentic_enhancement", {}) or {}
    market_cfg = load_config()
    paths = MarketSkillPaths()
    views_path = ROOT / agent_cfg["paths"]["agentic_views"]
    ep_path = ROOT / agent_cfg["paths"]["agentic_ep_views"]

    views = pd.read_parquet(views_path)
    cards = pd.read_parquet(ROOT / "data_processed" / "market_state_cards_daily_v1.parquet")
    mech = _mechanical_by_date(cards, market_cfg)

    lo = float(agent_cfg.get("confidence_min", 0.45))
    hi = float(agent_cfg.get("confidence_max", 0.75))
    agree_boost = float(enh.get("agree_confidence_boost", 0.03))
    raise_boost = float(enh.get("raise_confidence_boost", 0.06))
    lower_penalty = float(enh.get("lower_confidence_penalty", 0.06))
    override_min_conf = float(enh.get("new_override_min_confidence", 0.58))

    rows: list[dict[str, Any]] = []
    changed = 0
    dropped = 0
    for _, row in views.iterrows():
        item = row.to_dict()
        generator = str(item.get("generator", ""))
        if generator != "agentic_market_view_v1":
            rows.append(item)
            continue
        date = _date_key(item["date"])
        pair = (str(item.get("source_asset", "")), str(item.get("target_asset", "")))
        reverse = (pair[1], pair[0])
        candidates = mech.get(date, {})
        alignment = str(item.get("model_alignment", "mixed") or "mixed")
        conf0 = float(item.get("confidence", 0.0))
        conf = conf0
        tags = _json_loads(item.get("reason_tags", "[]")) if isinstance(item.get("reason_tags"), str) else item.get("reason_tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]

        if pair in candidates:
            base = candidates[pair]
            base_conf = float(base.get("confidence", conf0))
            base_tags = base.get("reason_tags", []) or []
            if isinstance(base_tags, str):
                base_tags = _json_loads(base_tags)
            if alignment == "raise_mechanical_confidence":
                conf = max(conf0, base_conf + raise_boost)
            elif alignment == "lower_mechanical_confidence":
                conf = min(conf0, base_conf - lower_penalty)
            else:
                conf = max(conf0, base_conf + agree_boost)
                if alignment in {"mixed", "", "deterministic_supervisor"}:
                    alignment = "agree_mechanical"
            tags = sorted(set([str(x) for x in tags] + [str(x) for x in base_tags] + ["mechanical_audit_calibrated"]))
        elif reverse in candidates:
            if alignment not in {"veto_mechanical_replacement", "new_agentic_override"} or conf0 < override_min_conf:
                dropped += 1
                continue
            tags = sorted(set([str(x) for x in tags] + ["agentic_veto_mechanical"]))
        else:
            if alignment not in {"new_agentic_override", "mixed"} or conf0 < override_min_conf:
                dropped += 1
                continue
            alignment = "new_agentic_override" if alignment == "mixed" else alignment
            tags = sorted(set([str(x) for x in tags] + ["agentic_new_override"]))

        conf = float(np.clip(conf, lo, hi))
        if abs(conf - conf0) > 1e-12 or alignment != str(item.get("model_alignment", "mixed") or "mixed"):
            changed += 1
        item["confidence"] = round(conf, 3)
        item["view_weight"] = round(conf, 3) if "view_weight" in item else item.get("view_weight")
        item["model_alignment"] = alignment
        item["reason_tags"] = json.dumps([str(x) for x in tags[:10]], ensure_ascii=False)
        rows.append(item)

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["date", "validator_status", "source_asset", "target_asset"]).reset_index(drop=True)
    backup = views_path.with_suffix(".pre_calibration.parquet")
    if not backup.exists():
        views.to_parquet(backup, index=False)
    out.to_parquet(views_path, index=False)
    ep = build_ep_views(out, paths=MarketSkillPaths(ep_views=ep_path))
    report = ROOT / "output" / "reports" / "agentic_view_calibration_report.md"
    report.write_text(
        "# Agentic View Calibration\n\n"
        f"- input_rows: {len(views)}\n"
        f"- output_rows: {len(out)}\n"
        f"- ep_views: {len(ep)}\n"
        f"- changed_agentic_rows: {changed}\n"
        f"- dropped_weak_override_rows: {dropped}\n"
        f"- backup: {backup.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    print(report.read_text(encoding="utf-8"))
    return out


if __name__ == "__main__":
    calibrate()
