"""Reuse the immutable V92 executor with an isolated coupon-aware book adapter.

Output locations and constructor dependency are overridden only in this process.
Original executor/source/data are never edited. Extra evidence hashes are checked
both before and after the full replay and registered in the new output only.
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import replay_v92_excluded_cb_universe as executor
from scripts import audit_v92_excluded_cb_universe as auditor
from src.v82_causal_cb_events import CausalQuantityBooks
from src.v93_coupon_events import augment_coupon_events

OUT = ROOT / "output/reports/cb_cashflows_v93"
DATA = ROOT / "data_processed/cb_cashflows_v93"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    bound = {}
    def bind(relative):
        path = ROOT / relative if isinstance(relative, str) else relative
        bound[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    manifest = json.loads(bind("data_raw/cb_coupons_v93/manifest.json").read_text(encoding="utf-8"))
    evidence = {r["ts_code"]: r for r in manifest}
    calendar = pd.read_parquet(bind("data_processed/cb_evidence_v50/last_known_balance_yuan.parquet"), columns=[]).index
    cases = [dict(doc="electric_2019", code="113008.SH", record="2019-02-01", ex="2019-02-11", pay="2019-02-15", unit=1.5,
                  tokens=["2019年2月1日", "2019年2月11日", "2019年2月15日", "1.5元"]),
             dict(doc="spdb_2024", code="110059.SH", record="2024-10-25", ex="2024-10-28", pay="2024-10-28", unit=3.2,
                  tokens=["2024年10月25日", "2024年10月28日", "3.20元"])]
    coupons = []
    for case in cases:
        proof = evidence[case["doc"]]
        if proof["status"] != "original_pdf_acquired_identity_checked":
            raise ValueError("Coupon evidence acquisition incomplete")
        pdf = bind(proof["pdf"])
        if bound[proof["pdf"]] != proof["sha256"]:
            raise ValueError("Coupon evidence changed")
        text = bind(pdf.with_suffix(".txt")).read_text(encoding="utf-8")
        compact = "".join(text.split())
        if not all(t in compact for t in [case["code"][:6]] + case["tokens"]):
            raise ValueError("Exact coupon schedule facts not found: " + case["doc"])
        known = calendar[calendar > pd.Timestamp(proof["publication_date"])][0]
        coupons.append(dict(event_id=f"reviewed_coupon_{case['code']}_{case['record']}", ts_code=case["code"], kind="coupon",
                            known_at=known, record_date=case["record"], ex_date=case["ex"], pay_date=case["pay"],
                            confirmed_at=pd.NaT, cash_yuan_per_bond=case["unit"], source=proof["url"],
                            confirmation_basis="schedule_only_cash_confirmation_missing", cash_basis="gross_before_investor_tax"))
    for relative in ["scripts/replay_v93_reviewed_coupon_schedules.py", "scripts/fetch_v93_coupon_schedules.py",
                     "src/v93_coupon_events.py", "tests/test_coupon_schedule_v93.py"]:
        bind(relative)
    books = []
    class CouponBook(CausalQuantityBooks):
        def __init__(self, names, codes, events, sessions, **kwargs):
            super().__init__(names, codes, augment_coupon_events(events, coupons, codes), sessions, **kwargs)
            books.append(self)
    executor.OUT, executor.DATA = OUT, DATA
    executor.CausalQuantityBooks = CouponBook
    executor.main()
    if len(books) != 1:
        raise ValueError("Unexpected executor book count")
    book = books[0]
    book.events.to_parquet(DATA / "reviewed_events.parquet", index=False)
    rights = []
    for row in pd.DataFrame(coupons).itertuples(index=False):
        quantities = book.entitlements[row.event_id]
        for name, qty in zip(book.names, quantities):
            rights.append(dict(event_id=row.event_id, ts_code=row.ts_code, asset_code=name, record_date=row.record_date,
                               ex_date=row.ex_date, scheduled_payment_date=row.pay_date, quantity=qty,
                               gross_claim_yuan=qty*row.cash_yuan_per_bond, confirmed_at=None,
                               actual_coupon_cash_yuan=0, source=row.source))
        if row.event_id in book.paid:
            raise ValueError("Schedule-only coupon was improperly paid")
    claims = pd.DataFrame(rights)
    claims.to_csv(OUT / "actual_registered_coupon_claims_NOT_receipts.csv", index=False, encoding="utf-8-sig")
    for relative, expected in bound.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Extra source changed during replay: " + relative)
    status_path = OUT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(version="v93_coupon_augmented_retrospective_screen", reviewed_coupon_schedules=len(coupons),
                  nonzero_coupon_basket_claims=int(claims.quantity.gt(1e-8).sum()),
                  gross_coupon_claim_yuan_NOT_cash=float(claims.gross_claim_yuan.sum()), confirmed_coupon_cash_yuan=0,
                  coupon_claims_not_investable_cash=True, source_sha256={**status["source_sha256"], **bound})
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    auditor.OUT, auditor.DATA = OUT, DATA
    auditor.main()
    print(claims.loc[claims.quantity.gt(1e-8), ["ts_code", "asset_code", "quantity", "gross_claim_yuan"]].to_string(index=False))


if __name__ == "__main__":
    main()
