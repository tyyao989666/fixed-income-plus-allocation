"""Bounded public CNInfo lookup; PDFs are evidence candidates, never auto-approved."""
import hashlib
import json
from pathlib import Path
import re
import time
import requests
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data_raw/terminal_announcements_v70'
CASES=[('110065.SH','淮矿转债','2021-06-01~2021-07-15'),
       ('113559.SH','永创转债','2021-06-01~2021-07-10'),
       ('113012.SH','骆驼转债','2021-08-01~2021-09-15'),
       ('123033.SZ','金力转债','2021-07-01~2021-09-10'),
       ('128050.SZ','钧达转债','2021-12-01~2022-02-15')]


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.cninfo.com.cn/'})
    log=[]
    for code,name,period in CASES:
        folder=RAW/code
        folder.mkdir(exist_ok=True)
        query=folder/'index.json'
        if not query.exists():
            response=session.post('https://www.cninfo.com.cn/new/hisAnnouncement/query',data=dict(
                pageNum=1,pageSize=30,column='sse' if code.endswith('.SH') else 'szse',tabName='fulltext',
                stock='',searchkey=name,secid='',plate='',category='',trade='',seDate=period,
                sortName='time',sortType='asc',isHLtitle='false'),timeout=30)
            response.raise_for_status()
            payload=response.json()
            query.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            time.sleep(.6)
        else:
            payload=json.loads(query.read_text(encoding='utf-8'))
        announcements=payload.get('announcements') or []
        chosen=[]
        for row in announcements:
            title=re.sub('<[^>]+>','',row.get('announcementTitle',''))
            if '赎回' not in title:
                continue
            if '结果' in title:
                chosen.append(('result',row,title))
            elif name in title and ('实施' in title or '赎回的公告' in title) and not any(x[0]=='schedule' for x in chosen):
                chosen.append(('schedule',row,title))
        chosen=chosen[:4]
        for kind,row,title in chosen:
            rel=row['adjunctUrl']
            if not re.fullmatch(r'finalpage/\d{4}-\d{2}-\d{2}/\d+\.[Pp][Dd][Ff]',rel):
                raise ValueError('Unexpected announcement path')
            url='https://static.cninfo.com.cn/'+rel
            path=folder/Path(rel).name
            if not path.exists():
                response=session.get(url,timeout=40)
                response.raise_for_status()
                if not response.content.startswith(b'%PDF'):
                    raise ValueError('Not a PDF')
                path.write_bytes(response.content)
                time.sleep(.4)
            pages=[p.extract_text() or '' for p in PdfReader(path).pages]
            compact=''.join(''.join(pages).split()).replace(',','').replace('，','')
            txt=path.with_suffix('.txt')
            txt.write_text('\n'.join(pages),encoding='utf-8')
            code_matched=code[:6] in compact
            fields={}
            for token in ['赎回登记日','赎回价格','赎回款发放日','到账日','最后交易日','兑付完毕','赎回兑付','本次赎回数量']:
                match=compact.find(token)
                if match>=0:
                    fields[token]=compact[max(0,match-15):match+170]
            item=dict(ts_code=code,name=name,kind=kind,title=title,url=url,publication_date=rel.split('/')[1],
                pdf=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                code_matched=code_matched,field_excerpts=fields,approved_for_ledger=False)
            log.append(item)
            print(json.dumps(item,ensure_ascii=False),flush=True)
    (RAW/'candidate_manifest.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
