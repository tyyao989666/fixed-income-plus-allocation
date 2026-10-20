"""Verify official notices and reconcile one exposure; no full-NAV certification."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import requests
from pypdf import PdfReader

RAW = ROOT / 'data_raw/cashflow_evidence_v68'
OUT = ROOT / 'output/reports/cashflow_repair_v68'
SOURCES = {
    'schedule': 'https://disc.static.szse.cn/download/disc/disk02/finalpage/2021-07-01/6b0b6065-09a3-4697-9ba8-f5fe03d9c718.PDF',
    'result': 'https://disc.static.szse.cn/disc/disk02/finalpage/2021-08-05/d68c113c-6442-4832-83f7-f46a8a971f34.PDF',
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name,url in SOURCES.items():
        path = RAW / (name+'.pdf')
        if not path.exists():
            response = requests.get(url, timeout=40)
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError('Unexpected announcement payload')
            path.write_bytes(response.content)
        pages = [p.extract_text() or '' for p in PdfReader(path).pages]
        compact = ''.join(''.join(pages).split()).replace(',','').replace('，','')
        required = ['128032','100.88','2021年7月27日','2021年8月4日']
        if name=='result':
            required += ['151091','15242060.08','已通过']
        else:
            required += ['2021年7月28日','100.70']
        if not all(s in compact for s in required):
            raise ValueError('Announcement facts not matched: '+name)
        evidence[name] = dict(url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            page_count=len(pages), fact_checks_passed=True)
    # Aggregate issuer cash check validates gross per-bond amount, not investor tax.
    assert abs(151091*100.88-15242060.08)<1e-6
    event = dict(ts_code='128032.SZ', name='双环转债', schedule_publication_date='2021-07-01',
        conservative_available_after_close='2021-07-02', result_publication_date='2021-08-05',
        record_date='2021-07-27', redemption_date='2021-07-28', payment_date='2021-08-04',
        delist_date='2021-08-05', gross_cash_yuan_per_bond=100.88,
        published_personal_fund_net_cash_yuan_per_bond=100.70,
        outstanding_bonds=151091, gross_paid_yuan=15242060.08,
        result_not_used_before_publication=True,
        cash_not_assumed_received_at_July_month_end=True,
        evidence=evidence, automatically_applied_to_history=False)
    (OUT / 'verified_shuanghuan_event.json').write_text(json.dumps(event,ensure_ascii=False,indent=2),encoding='utf-8')
    missing = pd.read_csv(OUT / 'security_month_missing_ledger.csv')
    row = missing[missing.ts_code.eq('128032.SZ') & missing.return_month.eq('2021-07-31')].iloc[0]
    quote_path = ROOT / 'data_processed/cb_evidence_v50/quotes/close_individual.parquet'
    volume_path = ROOT / 'data_processed/cb_evidence_v50/quotes/volume_individual.parquet'
    price = pd.read_parquet(quote_path)['128032.SZ']
    volume = pd.read_parquet(volume_path)['128032.SZ']
    signal = pd.Timestamp(row.signal_date)
    if abs(price.loc[signal]-row.previous_close)>1e-5:
        raise ValueError('Monthly/daily anchor price mismatch')
    tradable = price.gt(0)&volume.gt(0)
    candidates = price.index[(price.index > pd.Timestamp(event['conservative_available_after_close'])) &
        (price.index <= pd.Timestamp(event['record_date'])) & tradable]
    if not len(candidates):
        raise ValueError('No observed exit opportunity before termination')
    exit_date = candidates[0]
    qty = float(row.held_weight/row.previous_close)
    impact = dict(scope='one_base_TSM_position_only_not_full_strategy', signal_date=row.signal_date,
        opening_price=float(row.previous_close), opening_weight=float(row.held_weight),
        gross_redemption_price=100.88, gross_hold_to_redemption_return=float(100.88/row.previous_close-1),
        gross_principal_value_effect_NAV=qty*(100.88-row.previous_close),
        hold_to_redemption_is_scenario_not_verified_investor_action=True,
        illustrative_causal_exit_date=str(exit_date.date()), illustrative_exit_close=float(price.loc[exit_date]),
        exit_rule='first positive-volume local close after conservative schedule availability; T+1',
        illustrative_exit_return_after_30bp=float(price.loc[exit_date]*(1-.003)/row.previous_close-1),
        no_forced_redemption_if_sell_or_convert=True,
        schedule_only_used_for_exit_rule=True, result_only_used_for_ex_post_reconciliation=True,
        July_receivable_not_cash=qty*100.88, cash_payment_month='2021-08',
        funding_carry_taxes_and_coupon_ledger_not_fully_rebuilt=True,
        independent_quote_verification_complete=False, portfolio_return_repaired=False,
        quote_sha256=hashlib.sha256(quote_path.read_bytes()).hexdigest(),
        volume_sha256=hashlib.sha256(volume_path.read_bytes()).hexdigest())
    (OUT / 'shuanghuan_position_reconciliation.json').write_text(json.dumps(impact,indent=2),encoding='utf-8')
    print(json.dumps(impact,indent=2))


if __name__=='__main__':
    main()
