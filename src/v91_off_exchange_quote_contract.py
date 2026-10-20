"""Quarantine unproven OTC observations; transport success is not trade proof."""
import csv
import io
import json
import math
import re

import pandas as pd


def extract_answer(receipt):
    response = receipt.get("response", {})
    if response.get("ok") is not True:
        raise ValueError("Provider request not successful")
    blocks = response.get("data", {}).get("result", {}).get("content", [])
    answers = []
    for block in blocks:
        if block.get("type") == "text":
            payload = json.loads(block["text"])
            if payload.get("code") != 1:
                raise ValueError("Provider business response not successful")
            if payload.get("data", {}).get("answer"):
                answers.append(payload["data"])
    if len(answers) != 1:
        raise ValueError("Expected exactly one unambiguous answer table")
    return answers[0]


def number(cell):
    cell = cell.strip()
    if cell in {"", "--", "null", "None"}:
        return None
    try:
        value = float(cell.replace(",", ""))
    except ValueError as exc:
        raise ValueError("Unparseable numeric cell") from exc
    if not math.isfinite(value):
        raise ValueError("Nonfinite observed value")
    return value


def review_receipt(receipt, route, domestic_sessions):
    data = extract_answer(receipt)
    lines = [line.strip() for line in data["answer"].splitlines() if line.strip()]
    if not lines or not all(line.startswith("|") and line.endswith("|") for line in lines):
        raise ValueError("Unexpected narrative or truncated markdown table")
    cells = [[c.strip() for c in row[1:-1]] for row in csv.reader(lines, delimiter="|")]
    header = cells[0]
    if len(cells) < 3 or len(cells[1]) != len(header) or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells[1]):
        raise ValueError("Invalid markdown table separator")
    if len(set(header)) != len(header):
        raise ValueError("Duplicate columns")
    price_headers = [c for c in header if c.startswith("收盘价")]
    if len(price_headers) != 1 or not {"证券代码", "证券简称", "日期", "成交量"}.issubset(header):
        raise ValueError("Required identity/date/price/volume schema absent")
    params = data.get("indicators_params", {})
    volume_unit = params.get("成交量", {}).get("单位设置")
    date_settings = {params.get(k, {}).get("交易日期") for k in ["收盘价", "成交量"]}
    sessions = set(pd.DatetimeIndex(domestic_sessions).normalize())
    rows, seen = [], set()
    for values in cells[2:]:
        if len(values) != len(header):
            raise ValueError("Ragged table")
        raw = dict(zip(header, values))
        if not re.fullmatch(r"\d{8}", raw["日期"]):
            raise ValueError("Invalid date format")
        day = pd.to_datetime(raw["日期"], format="%Y%m%d", errors="raise")
        if day in seen:
            raise ValueError("Duplicate security/date")
        seen.add(day)
        if not pd.Timestamp(receipt["start"]) <= day <= pd.Timestamp(receipt["end"]):
            raise ValueError("Observation outside requested interval")
        price, volume = number(raw[price_headers[0]]), number(raw["成交量"])
        reasons = []
        if raw["证券代码"] != route["new_code"] + ".NQ":
            reasons.append("security_identity_mismatch")
        if day not in sessions:
            reasons.append("not_domestic_session")
        if day < max(pd.Timestamp(route["known_at"]), pd.Timestamp(route["transfer_start"])):
            reasons.append("before_visible_transfer_start")
        if day.weekday() not in route["scheduled_weekdays"]:
            reasons.append("not_initial_scheduled_transfer_day")
        if price is None or price <= 0:
            reasons.append("invalid_close")
        if volume is None:
            reasons.append("missing_quantity")
        elif volume <= 0:
            reasons.append("no_positive_quantity")
        if volume_unit not in {"张", "手"}:
            reasons.append("quantity_unit_not_verified")
        if "最新" in date_settings:
            reasons.append("indicator_latest_date_metadata_requires_review")
        # Even positive samples need matched OTC trade/lot/rights evidence.
        reasons.append("otc_trade_and_custody_evidence_not_verified")
        rows.append(dict(date=day, old_code=route["old_code"], vendor_code=raw["证券代码"],
                         vendor_name=raw["证券简称"], observed_close=price, observed_volume=volume,
                         claimed_volume_unit=volume_unit, reasons=";".join(reasons),
                         review_status="quarantined", execution_allowed=False,
                         approved_for_nav=False, approved_for_returns=False))
    if len(cells) < 3:
        raise ValueError("Empty table")
    return pd.DataFrame(rows)


def default_state_at_signal(code, signal_date, notices):
    """A published nonpayment notice is a credit flag, not a zero recovery mark."""
    day = pd.Timestamp(signal_date)
    if pd.isna(day):
        raise ValueError("Signal date required")
    visible = []
    for row in notices:
        if row["old_code"] != code:
            continue
        known = pd.Timestamp(row["known_at"])
        if pd.isna(known) or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
            raise ValueError("Dated evidence required")
        if known <= day:
            visible.append(row)
    return dict(nonpayment_notice_visible=bool(visible),
                auto_par_redemption_allowed=False, recovery_value_yuan=None,
                actual_cash_received_yuan=None, historical_trades_changed=False)
