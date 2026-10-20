"""One fixed four-event overlay, not a search for profitable historical exits."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import replay_v95_excluded_cb_base as executor
from scripts import audit_v92_excluded_cb_universe as auditor
from src.v82_causal_cb_events import CausalQuantityBooks
from src.v94_distress_exit import adjust_targets, visible_exit_mask

OUT = ROOT / "output/reports/cb_cashflows_v95"
DATA = ROOT / "data_processed/cb_cashflows_v95"


def main():
    for folder in [OUT, DATA]:
        folder.mkdir(parents=True, exist_ok=True)
    bound = {}
    def bind(relative):
        path = ROOT / relative if isinstance(relative, str) else relative
        bound[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    parent = json.loads(bind("output/reports/cb_cashflows_v93/status.json").read_text(encoding="utf-8"))
    for relative, expected in parent["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Parent source changed: " + relative)
        bound[relative] = expected
    bind("scripts/replay_v95_excluded_cb_base.py")
    bind("config/cb_research_exclusions_v95.json")
    policy = json.loads(bind("config/v94_distress_exit_protocol.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(bind("data_processed/cb_cashflows_v93/reviewed_events.parquet"))
    calendar = pd.read_parquet(bind("data_processed/cb_cashflows_v92/signal_buy_eligibility.parquet"), columns=[]).index
    notes = json.loads(bind("data_raw/distress_triggers_v94/trigger_manifest.json").read_text(encoding="utf-8"))
    delist = json.loads(bind("data_raw/delisting_trigger_v94/manifest.json").read_text(encoding="utf-8"))[0]
    if delist["status"] != "original_pdf_acquired_identity_checked":
        raise ValueError("Delisting primary evidence incomplete")
    notes.append(dict(ts_code=delist["ts_code"], publication_date=delist["publication_date"],
                      trigger_type="exchange_decides_security_delisting", title=delist["title"] if "title" in delist else "Issuer delisting decision notice",
                      url=delist["url"], pdf=delist["pdf"], sha256=delist["sha256"]))
    if {r["ts_code"] for r in notes} != {"128114.SZ", "113578.SH", "127033.SZ", "123099.SZ"}:
        raise ValueError("Unexpected repair-case scope")
    for row in notes:
        if row["trigger_type"] not in policy["trigger_types"]:
            raise ValueError("Trigger not in frozen rule")
        pdf = bind(row["pdf"])
        if bound[row["pdf"]] != row["sha256"]:
            raise ValueError("Trigger PDF changed")
        text = bind(pdf.with_suffix(".txt")).read_text(encoding="utf-8")
        if row["ts_code"][:6] not in text:
            raise ValueError("Primary bond identifier absent")
        row["known_at"] = str(calendar[calendar > pd.Timestamp(row["publication_date"])][0].date())
    for relative in ["scripts/replay_v95_six_cb_exclusions.py", "scripts/fetch_v94_distress_triggers.py",
                     "scripts/fetch_v94_delisting_trigger.py", "src/v94_distress_exit.py", "tests/test_distress_exit_v94.py"]:
        bind(relative)
    (OUT / "reviewed_trigger_rules.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    log, books = [], []
    class GuardedBook(CausalQuantityBooks):
        def __init__(self, names, codes, original_events, sessions, **kwargs):
            if not set(original_events.event_id).issubset(events.event_id):
                raise ValueError("Incompatible base event registry")
            super().__init__(names, codes, events, sessions, **kwargs)
            books.append(self)

        def step(self, day, target, buy_allowed, price, executable, valuation, *, signal_date):
            revised, permission, blocked = adjust_targets(target, buy_allowed, self.codes, signal_date, notes)
            truncated = [r for r in notes if pd.Timestamp(r["known_at"]) <= pd.Timestamp(signal_date)]
            expected = visible_exit_mask(self.codes, signal_date, truncated)
            np.testing.assert_array_equal(blocked, expected)
            previous = self.quantity.copy()
            result = super().step(day, revised, permission, price, executable, valuation, signal_date=signal_date)
            if (result["fills"][:, blocked] > 1e-10).any():
                raise ValueError("Risk-blocked security purchased")
            for c in np.flatnonzero(blocked):
                for a, name in enumerate(self.names):
                    if previous[a, c] > 1e-8 or self.quantity[a, c] > 1e-8:
                        log.append(dict(date=day, signal_date=signal_date, asset_code=name, ts_code=self.codes[c],
                                        quantity_before=previous[a, c], quantity_after=self.quantity[a, c],
                                        fill_quantity=result["fills"][a, c], observed_close=price[c],
                                        executable_flag=bool(executable[c]), unavailable_target=bool(np.isclose(revised[a].sum(), 0))))
            return result
    executor.OUT, executor.DATA = OUT, DATA
    executor.CausalQuantityBooks = GuardedBook
    executor.main()
    pd.DataFrame(log).to_csv(OUT / "risk_exit_inventory_actions.csv", index=False, encoding="utf-8-sig")
    if len(books) != 1:
        raise ValueError("Unexpected book count")
    books[0].events.to_parquet(DATA / "reviewed_events.parquet", index=False)
    eligibility = pd.read_parquet(DATA / "signal_buy_eligibility.parquet")
    for row in notes:
        eligibility.loc[eligibility.index >= pd.Timestamp(row["known_at"]), row["ts_code"]] = False
    eligibility.to_parquet(DATA / "risk_adjusted_signal_buy_eligibility.parquet")
    for relative, expected in bound.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Extra source changed during replay: " + relative)
    status_path = OUT / "status.json"
    state = json.loads(status_path.read_text(encoding="utf-8"))
    state.update(version="v95_six_cb_exclusions_research", event_risk_overlay_enabled=True,
                 reviewed_distress_triggers=len(notes), scope_not_exhaustive_all_issuer_events=True,
                 same_coupon_basis_as_v93=True, daily_risk_future_truncation_checks=len(calendar)-1,
                 execution_liquidity_capacity_certified=False, formal_promotion=False,
                 source_sha256={**state["source_sha256"], **bound})
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    auditor.OUT, auditor.DATA = OUT, DATA
    auditor.main()
    print(pd.DataFrame(log).to_string(index=False))


if __name__ == "__main__":
    main()
