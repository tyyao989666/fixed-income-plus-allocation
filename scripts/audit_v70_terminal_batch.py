"""Independent artifact/prefix review; event closure is NOT portfolio certification."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v69_terminal_position_book import TerminalPositionBook

OUT=ROOT/'output/reports/terminal_batch_v70'


def main():
    status=json.loads((OUT/'status.json').read_text())
    for name,digest in status['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    reviewed=json.loads((OUT/'reviewed_events.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(OUT/'seven_position_reconciliation.csv')
    daily=pd.read_csv(OUT/'daily_position_ledger.csv')
    close=pd.read_parquet(ROOT/'data_processed/cb_evidence_v50/quotes/close_individual.parquet')
    volume=pd.read_parquet(ROOT/'data_processed/cb_evidence_v50/quotes/volume_individual.parquet')
    checks=[]
    for row in summary.itertuples(index=False):
        pair=reviewed[row.ts_code]
        schedule,result=pair['schedule'],pair['result']
        frame=daily[daily.ts_code.eq(row.ts_code)&daily['mode'].eq(row.mode)].copy()
        start=pd.Timestamp(row.opening_date)
        assert (pd.to_datetime(frame.date)>start).all()
        assert frame.receipt.gt(0).sum()<=1
        assert not ((frame.sale_quantity>0)&(~frame.schedule_known)).any()
        assert frame.loc[~frame.result_known,'receipt'].eq(0).all()
        assert (frame.cash>=0).all() and (frame.receivable>=0).all() and (frame.quantity>=0).all()
        assert np.isclose(frame.cash.iloc[-1],row.ending_cash)
        # Remove the future result completely: every pre-publication row must
        # match, including claims and requested/filled trades.
        no_result=TerminalPositionBook(row.opening_weight/row.opening_price,row.opening_price,start,schedule,None)
        prefix=frame[pd.to_datetime(frame.date)<=pd.Timestamp(result['result_publication_date'])]
        for day in pd.to_datetime(prefix.date):
            p,v=close.at[day,row.ts_code],volume.at[day,row.ts_code]
            actual=no_result.step(day,float(p) if pd.notna(p) else None,float(v) if pd.notna(v) else None,
                                  row.mode=='schedule_exit')
            expected=frame[frame.date.eq(str(day.date()))].iloc[0]
            for key in ['quantity','cash','receivable','provisional_nav','receipt','cost','sale_quantity']:
                assert np.isclose(actual[key],expected[key],rtol=1e-10,atol=1e-14),(row.ts_code,day,key)
        # Settlement changes liquidity, not total wealth, and never pays again.
        paid=np.flatnonzero(frame.receipt.to_numpy()>0)
        for i in paid:
            assert i>0
            assert np.isclose(frame.provisional_nav.iloc[i],frame.provisional_nav.iloc[i-1])
            assert np.isclose(frame.cash.iloc[i]-frame.cash.iloc[i-1],frame.receipt.iloc[i])
            assert np.isclose(frame.receivable.iloc[i-1]-frame.receivable.iloc[i],frame.receipt.iloc[i])
        checks.append(dict(ts_code=row.ts_code,mode=row.mode,prefix_dates=len(prefix),
            future_result_invariance=True,settlement_conservation=True,no_double_payment=True))
    pd.DataFrame(checks).to_csv(OUT/'independent_path_checks.csv',index=False)
    basic=pd.read_parquet(ROOT/'data_raw/tushare/cb_basic_all.parquet').drop_duplicates('ts_code')
    queue=pd.read_csv(OUT/'updated_repair_queue.csv')
    scope=queue[queue.asset.eq('cb')].merge(basic[['ts_code','cb_type','bond_short_name']],on='ts_code',how='left',validate='one_to_one')
    scope['taxonomy_review_only_no_historical_exclusion']=True
    scope.to_csv(OUT/'cb_vs_exchangeable_scope.csv',index=False,encoding='utf-8-sig')
    types=scope.groupby('cb_type',dropna=False).agg(securities=('ts_code','size'),weight_months=('summed_weight_months','sum'))
    types.to_csv(OUT/'security_type_summary.csv')
    fetch_status=[]
    for path in (ROOT/'data_raw/terminal_announcements_v70').glob('*/index.json'):
        index=json.loads(path.read_text(encoding='utf-8'))
        complete=not index.get('hasMore',False) and len(index.get('announcements')or[])==index.get('totalAnnouncement')
        assert complete,'Unfetched source pages: '+path.parent.name
        fetch_status.append(dict(ts_code=path.parent.name,index_complete=True,
            announcements=index['totalAnnouncement'],index_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(fetch_status).to_csv(OUT/'announcement_index_audit.csv',index=False)
    result=dict(passed=True,paths=len(checks),future_result_prefix_dates=sum(x['prefix_dates'] for x in checks),
        chronology_and_conservation_passed=True,source_hashes_passed=True,index_pages_complete=True,
        source_taxonomy_counts=types.securities.to_dict(),taxonomy_used_as_trading_filter=False,
        full_portfolio_certified=False,formal_promotion=False,
        artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.csv','.json'] and p.name!='artifact_audit.json'})
    (OUT/'artifact_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
