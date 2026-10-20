"""Reviewed source acquisition plus isolated daily replay of three event positions."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from bs4 import BeautifulSoup
from src.v69_terminal_position_book import TerminalPositionBook

RAW=ROOT/'data_raw/cashflow_evidence_v69'
OUT=ROOT/'output/reports/terminal_books_v69'
SOURCES={
    'oppein_schedule': ('https://www.oppein.com/upfile/2021/07/20210728153101_819.pdf',
        ['113543','100.543','2021年7月9日','2021年7月12日']),
    'oppein_result': ('https://www.oppein.com/upfile/2021/07/20210728163917_243.pdf',
        ['113543','100.543','14000','1407602','兑付完毕']),
    'yongchuang_schedule': ('https://static.cninfo.com.cn/finalpage/2021-06-16/1210238874.PDF',
        ['113559','100.405','2021年6月25日','2021年6月28日']),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    evidence={}
    for name,(url,tokens) in SOURCES.items():
        path=RAW/(name+'.pdf')
        if not path.exists():
            response=requests.get(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.oppein.com/'},timeout=45)
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError('Not an announcement PDF: '+name)
            path.write_bytes(response.content)
        reader=PdfReader(path)
        compact=''.join(''.join(p.extract_text() or '' for p in reader.pages).split()).replace(',','').replace('，','')
        if not all(t in compact for t in tokens):
            raise ValueError('Source fact check failed: '+name)
        evidence[name]=dict(url=url,sha256=sha(path),verified_tokens=tokens,pages=len(reader.pages))
    # Save issuer's dated index page as separate publication-date evidence.
    index_url='https://www.oppein.com/wap/index.php?ac=article&at=list&page=103&tid=19'
    index_path=RAW/'oppein_dated_index.html'
    if not index_path.exists():
        response=requests.get(index_url,headers={'User-Agent':'Mozilla/5.0'},timeout=40)
        response.raise_for_status()
        index_path.write_bytes(response.content)
    html=index_path.read_text(encoding='utf-8',errors='replace')
    if '07-01' not in html or '20210728153101_819.pdf' not in html:
        raise ValueError('Issuer dated schedule index not matched')
    evidence['oppein_dated_index']=dict(url=index_url,sha256=sha(index_path))
    result_index_url='https://www.oppein.com/wap/index.php?ac=article&at=list&page=101&tid=19'
    result_index_path=RAW/'oppein_dated_result_index.html'
    if not result_index_path.exists():
        response=requests.get(result_index_url,headers={'User-Agent':'Mozilla/5.0'},timeout=40)
        response.raise_for_status()
        result_index_path.write_bytes(response.content)
    soup=BeautifulSoup(result_index_path.read_bytes(),'html.parser')
    matched=[a for a in soup.find_all('a') if a.get('href','').endswith('20210728163917_243.pdf')]
    if len(matched)!=1 or '07-13' not in matched[0].parent.parent.get_text(' ',strip=True):
        raise ValueError('Historical result publication date not matched')
    evidence['oppein_dated_result_index']=dict(url=result_index_url,sha256=sha(result_index_path),publication_date='2021-07-13')
    shuang_path=ROOT/'output/reports/cashflow_repair_v68/verified_shuanghuan_event.json'
    shuang=json.loads(shuang_path.read_text(encoding='utf-8'))
    for name, item in shuang['evidence'].items():
        assert sha(ROOT/'data_raw/cashflow_evidence_v68'/(name+'.pdf'))==item['sha256']
    oppein=dict(ts_code='113543.SH',record_date='2021-07-09',redemption_date='2021-07-12',
        payment_date='2021-07-12',conservative_available_after_close='2021-07-02',gross_cash_yuan_per_bond=100.543)
    # The issuer's dated index proves July13; PDF upload path July28 is not the
    # publication date. Cash confirmation remains delayed until the next session.
    oppein_result=dict(oppein,result_publication_date='2021-07-13')
    yong=dict(ts_code='113559.SH',record_date='2021-06-25',redemption_date='2021-06-28',
        payment_date='2021-06-28',conservative_available_after_close='2021-06-17',gross_cash_yuan_per_bond=100.405)
    cases=[(shuang,shuang),(oppein,oppein_result),(yong,None)]
    assert abs(14000*100.543-1407602)<1e-6
    qpath=ROOT/'data_processed/cb_evidence_v50/quotes/close_individual.parquet'
    vpath=ROOT/'data_processed/cb_evidence_v50/quotes/volume_individual.parquet'
    close,volume=pd.read_parquet(qpath),pd.read_parquet(vpath)
    missing_path=ROOT/'output/reports/cashflow_repair_v68/security_month_missing_ledger.csv'
    gaps=pd.read_csv(missing_path)
    daily,summary=[],[]
    for schedule,result in cases:
        code=schedule['ts_code']
        gap=gaps[gaps.ts_code.eq(code)].iloc[0]
        start=pd.Timestamp(gap.signal_date)
        price,weight=float(gap.previous_close),float(gap.held_weight)
        if not np.isclose(close.at[start,code],price,rtol=0,atol=1e-5):
            raise ValueError('Opening price basis mismatch: '+code)
        quantity=weight/price
        for mode in ['hold_unconverted','schedule_exit']:
            book=TerminalPositionBook(quantity,price,start,schedule,result)
            rows=[]
            for day in close.index[(close.index>start)&(close.index<=pd.Timestamp('2021-08-10'))]:
                p=close.at[day,code]
                v=volume.at[day,code]
                row=book.step(day,float(p) if pd.notna(p) else None,
                    float(v) if pd.notna(v) else None,mode=='schedule_exit')
                row.update(ts_code=code,mode=mode)
                rows.append(row)
            frame=pd.DataFrame(rows)
            daily.extend(rows)
            assert frame.receipt.gt(0).sum()<=1
            assert frame.loc[~frame.schedule_known,'sale_quantity'].eq(0).all()
            assert frame.loc[pd.to_datetime(frame.date)<=pd.Timestamp(schedule['record_date']),'receipt'].eq(0).all()
            if mode=='hold_unconverted':
                first=frame[frame.cancelled].iloc[0]
                assert np.isclose(first.receivable+first.receipt,quantity*schedule['gross_cash_yuan_per_bond'])
            sales=frame[frame.sale_quantity>0]
            final=frame.iloc[-1]
            summary.append(dict(ts_code=code,mode=mode,opening_date=str(start.date()),opening_weight=weight,
                opening_price=price,exit_date=sales.iloc[0].date if len(sales) else None,
                ending_cash=float(final.cash),ending_receivable=float(final.receivable),
                ending_quantity=float(final.quantity),ending_value=float(final.provisional_nav),
                cumulative_cost=float(frame.cost.sum()),cash_receipt_count=int(frame.receipt.gt(0).sum()),
                missing_mark_days=int(frame.missing_held_mark.sum()),
                value_change_NAV_bp=float((final.provisional_nav-weight)*10000),
                schedule_source_verified=True,result_document_verified=result is not None,
                settlement_recognition='delayed_until_result_visible_NOT_actual_bank_receipt_time',
                formal_portfolio=False))
    pd.DataFrame(daily).to_csv(OUT/'daily_position_ledger.csv',index=False)
    table=pd.DataFrame(summary)
    table.to_csv(OUT/'three_position_reconciliation.csv',index=False)
    reviewed={s['ts_code']:dict(schedule=s,result=r) for s,r in cases}
    (OUT/'reviewed_events.json').write_text(json.dumps(reviewed,ensure_ascii=False,indent=2),encoding='utf-8')
    queue=pd.read_csv(ROOT/'output/reports/cashflow_repair_v68/repair_priority_by_security.csv')
    queue['schedule_document_verified']=queue.ts_code.isin(reviewed)
    queue['result_document_verified']=queue.ts_code.isin([s['ts_code'] for s,r in cases if r is not None])
    queue['full_security_return_history_certified']=False
    queue.to_csv(OUT/'updated_repair_queue.csv',index=False)
    status=dict(evidence=evidence,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
        [Path(__file__),ROOT/'src/v69_terminal_position_book.py',qpath,vpath,missing_path,shuang_path]},
        cases=3,paths=6,parameter_search=False,formal_promotion=False,
        original_arrays_modified=False,full_backtest_run=False,
        schedule_verified_positions=3,result_verified_positions=2,
        unresolved_highest_priority='110065.SH primary schedule/result still needed',
        end_date='2021-08-10',cash_interest_included=False,coupons_complete=False,
        fees='30bp risky sale; opening inventory inherits historical weight, entry cost not recharged',
        quote_independent_validation=False,
        limitation='isolated inherited positions, not a reinvested full portfolio; fractional quantities, no capacity certification')
    (OUT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(table[['ts_code','mode','exit_date','ending_cash','ending_receivable','value_change_NAV_bp','cash_receipt_count']].to_string(index=False))


if __name__=='__main__':
    main()
