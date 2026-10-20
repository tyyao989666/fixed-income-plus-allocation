"""Approve exact source-backed events and replay inherited positions without refits."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from pypdf import PdfReader
from src.v69_terminal_position_book import TerminalPositionBook
from src.v70_observation_dates import month_end_session

OUT=ROOT/'output/reports/terminal_batch_v70'
RAW=ROOT/'data_raw/terminal_announcements_v70'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def text(path):
    return ''.join(''.join(p.extract_text() or '' for p in PdfReader(path).pages).split()).replace(',','').replace('，','')


def chinese_date(s):
    t=pd.Timestamp(s)
    return f'{t.year}年{t.month}月{t.day}日'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg_path=ROOT/'config/v70_reviewed_terminal_events.json'
    cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
    manifest_path=RAW/'candidate_manifest.json'
    candidates=json.loads(manifest_path.read_text(encoding='utf-8'))
    parent_status=json.loads((ROOT/'output/reports/terminal_books_v69/status.json').read_text(encoding='utf-8'))
    for name,digest in parent_status['source_hashes'].items():
        assert sha(ROOT/name)==digest, name
    inherited_path=ROOT/'output/reports/terminal_books_v69/reviewed_events.json'
    inherited=json.loads(inherited_path.read_text(encoding='utf-8'))
    qpath=ROOT/'data_processed/cb_evidence_v50/quotes/close_individual.parquet'
    vpath=ROOT/'data_processed/cb_evidence_v50/quotes/volume_individual.parquet'
    gap_path=ROOT/'output/reports/cashflow_repair_v68/security_month_missing_ledger.csv'
    close,volume,gaps=pd.read_parquet(qpath),pd.read_parquet(vpath),pd.read_csv(gap_path)
    dates=close.index
    reviewed={k:v for k,v in inherited.items() if k in ['128032.SZ','113543.SH']}
    evidence_rows=[]
    bound={str(p.relative_to(ROOT)):sha(p) for p in [cfg_path,manifest_path,inherited_path,Path(__file__),
        ROOT/'src/v69_terminal_position_book.py',ROOT/'src/v70_observation_dates.py',qpath,vpath,gap_path]}
    for event in cfg['events']:
        code=event['ts_code']
        pair={}
        for kind in ['schedule','result']:
            key=event[kind+'_id']
            matches=[x for x in candidates if x['ts_code']==code and x['kind']==kind and Path(x['pdf']).stem==key]
            if len(matches)!=1:
                raise ValueError('Ambiguous/missing reviewed document: '+code+' '+kind)
            item=matches[0]
            path=ROOT/item['pdf']
            if sha(path)!=item['sha256']:
                raise ValueError('Source PDF hash changed')
            compact=text(path)
            required=[code[:6],str(event['gross_cash_yuan_per_bond'])]
            if kind=='schedule':
                required += [chinese_date(event['record_date']),chinese_date(event['payment_date'])]
            else:
                required += [str(event['bonds_paid']),format(event['gross_total_paid'],'.2f').rstrip('0').rstrip('.')]
                required += event['result_tokens']
            if not all(token in compact for token in required):
                raise ValueError('Reviewed facts absent: '+code+' '+kind)
            bound[str(path.relative_to(ROOT))]=sha(path)
            evidence_rows.append(dict(ts_code=code,kind=kind,url=item['url'],publication_date=item['publication_date'],
                pdf=item['pdf'],sha256=item['sha256'],reviewed=True,review_note=event['review_note']))
            pair[kind]=item
        assert np.isclose(event['bonds_paid']*event['gross_cash_yuan_per_bond'],event['gross_total_paid'],rtol=0,atol=.005)
        publication=pd.Timestamp(pair['schedule']['publication_date'])
        available=dates[dates>publication][0]
        schedule={k:event[k] for k in ['ts_code','record_date','redemption_date','payment_date','last_trade_date','gross_cash_yuan_per_bond']}
        schedule.update(conservative_available_after_close=str(available.date()),schedule_publication_date=str(publication.date()))
        result=dict(schedule,result_publication_date=pair['result']['publication_date'])
        reviewed[code]=dict(schedule=schedule,result=result)
    rows,summary=[],[]
    for code,pair in reviewed.items():
        schedule,result=pair['schedule'],pair['result']
        gap=gaps[gaps.ts_code.eq(code)].iloc[0]
        start=month_end_session(gap.signal_date,dates)
        weight,price=float(gap.held_weight),float(gap.previous_close)
        assert np.isclose(close.at[start,code],price,atol=1e-5,rtol=0)
        stop=max(pd.Timestamp(schedule['payment_date']),pd.Timestamp(result['result_publication_date']))+pd.Timedelta(days=10)
        for mode in ['hold_unconverted','schedule_exit']:
            book=TerminalPositionBook(weight/price,price,start,schedule,result)
            daily=[]
            for day in dates[(dates>start)&(dates<=stop)]:
                p,v=close.at[day,code],volume.at[day,code]
                row=book.step(day,float(p) if pd.notna(p) else None,float(v) if pd.notna(v) else None,mode=='schedule_exit')
                row.update(ts_code=code,mode=mode)
                daily.append(row)
            frame=pd.DataFrame(daily)
            rows.extend(daily)
            sales=frame[frame.sale_quantity>0]
            payments=frame[frame.receipt>0]
            final=frame.iloc[-1]
            assert len(payments)<=1
            assert frame.loc[~frame.schedule_known,'sale_quantity'].eq(0).all()
            if len(payments):
                assert pd.Timestamp(payments.iloc[0].date)>pd.Timestamp(result['result_publication_date'])
                assert pd.Timestamp(payments.iloc[0].date)>=pd.Timestamp(schedule['payment_date'])
            if mode=='hold_unconverted':
                assert final.quantity==0 and final.receivable==0
                assert np.isclose(final.cash,weight/price*schedule['gross_cash_yuan_per_bond'])
            elif len(sales):
                assert not len(payments), 'Sold inventory collected redemption twice'
                assert final.quantity==0 and final.receivable==0
            summary.append(dict(ts_code=code,mode=mode,opening_date=str(start.date()),opening_month_label=gap.signal_date,opening_weight=weight,
                opening_price=price,schedule_known_before_formation=pd.Timestamp(schedule['conservative_available_after_close'])<start,
                exit_date=sales.iloc[0].date if len(sales) else None,
                receipt_date=payments.iloc[0].date if len(payments) else None,
                payment_count=len(payments),ending_cash=float(final.cash),ending_receivable=float(final.receivable),
                ending_quantity=float(final.quantity),value_change_NAV_bp=float((final.provisional_nav-weight)*10000),
                missing_mark_days=int(frame.missing_held_mark.sum()),formal_portfolio=False))
    table=pd.DataFrame(summary)
    table.to_csv(OUT/'seven_position_reconciliation.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'daily_position_ledger.csv',index=False)
    pd.DataFrame(evidence_rows).to_csv(OUT/'reviewed_source_register.csv',index=False,encoding='utf-8-sig')
    (OUT/'reviewed_events.json').write_text(json.dumps(reviewed,ensure_ascii=False,indent=2),encoding='utf-8')
    queue=pd.read_csv(ROOT/'output/reports/cashflow_repair_v68/repair_priority_by_security.csv')
    queue['schedule_document_verified']=queue.ts_code.isin(reviewed)
    queue['result_document_verified']=queue.ts_code.isin(reviewed)
    queue['full_security_return_history_certified']=False
    queue.to_csv(OUT/'updated_repair_queue.csv',index=False)
    cb=queue[queue.asset.eq('cb')]
    covered=cb[cb.ts_code.isin(reviewed)]
    status=dict(source_hashes=bound,reviewed_events=len(reviewed),replay_paths=len(table),
        reviewed_new_pdf_count=len(evidence_rows),cash_and_receivables_closed_all_paths=bool((table.ending_receivable==0).all()),
        covered_base_missing_security_months=int(covered.missing_months.sum()),
        total_base_missing_security_months=int(cb.missing_months.sum()),
        covered_base_weight_month_share=float(covered.summed_weight_months.sum()/cb.summed_weight_months.sum()),
        all_paths_checked_no_double_payment=True,parameter_search=False,full_portfolio_replay=False,
        formal_promotion=False,training_run=False,original_data_changed=False,
        cash_interest_and_other_coupons_complete=False,quote_independently_certified=False,
        limitations=['Seven inherited positions only; not a reinvested full portfolio',
            'Legacy formation exposures may predate or postdate known termination; no retroactive universe rewriting',
            'Return labels and PPO rewards remain blocked pending full cashflow certification'])
    (OUT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(table[['ts_code','mode','exit_date','receipt_date','schedule_known_before_formation','value_change_NAV_bp']].to_string(index=False))
    print(json.dumps({k:v for k,v in status.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':
    main()
