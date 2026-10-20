"""Schedule-only T+1 execution controls, isolated from the unchanged replay policy."""
import re
import pandas as pd

SCHEDULE_FIELDS = ["ts_code", "known_at", "last_trade_date", "source", "schedule_sha256"]


def terminal_controls(codes, schedules, signal_date, execution_date, sessions):
    codes = list(codes)
    calendar = pd.DatetimeIndex(sessions)
    signal, execution = pd.Timestamp(signal_date), pd.Timestamp(execution_date)
    if (len(codes) != len(set(codes)) or calendar.has_duplicates
            or not calendar.is_monotonic_increasing or execution not in calendar):
        raise ValueError("Explicit unique securities and ordered execution calendar required")
    position = calendar.get_loc(execution)
    if position == 0 or calendar[position - 1] != signal:
        raise ValueError("Schedule instruction requires previous domestic close")
    if set(SCHEDULE_FIELDS) - set(schedules.columns):
        raise ValueError("Incomplete source-bound schedule")
    # Payment amounts, confirmation fields and future returns are not policy inputs.
    frame = schedules[SCHEDULE_FIELDS].copy()
    if frame.ts_code.duplicated().any() or not frame.ts_code.isin(codes).all():
        raise ValueError("Ambiguous schedule revision or unknown security")
    for column in ["known_at", "last_trade_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError("Unknown schedule timing")
    if (frame.source.isna().any() or frame.source.astype(str).str.strip().eq("").any()
            or not frame.schedule_sha256.map(lambda h: isinstance(h, str)
                and re.fullmatch(r"[0-9a-f]{64}", h) is not None).all()):
        raise ValueError("Schedule provenance required")
    output = pd.DataFrame(index=pd.Index(codes, name="ts_code"))
    for field in ["schedule_visible", "block_new_buy", "request_liquidation",
                  "block_execution", "needs_frozen_inventory_review"]:
        output[field] = False
    for row in frame.loc[frame.known_at.le(signal)].itertuples(index=False):
        output.loc[row.ts_code, ["schedule_visible", "block_new_buy"]] = True
        if execution <= row.last_trade_date:
            output.at[row.ts_code, "request_liquidation"] = True
        else:
            output.loc[row.ts_code, ["block_execution", "needs_frozen_inventory_review"]] = True
    return output
