"""Download only provider-returned thsi CSVs via verified HTTPS; no ingestion."""
import io
import json
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_raw/boundary_v66'


def main():
    rows = []
    for name in ['dividend_history', 'treasury_history', 'equity_close_history']:
        dest = OUT / (name+'.csv')
        if dest.exists():
            continue
        payload = json.loads((OUT / (name+'.json')).read_text(encoding='utf-8'))
        blocks = payload['response']['data']['result']['content']
        data = json.loads(next(b['text'] for b in blocks if b.get('type', 'text') == 'text'))['data']
        links = re.findall(r'https?://[^\s<>"\u3002]+?\.csv', data['answer'])
        if len(links) != 1:
            raise ValueError('Expected one structured vendor CSV link')
        parsed = urlparse(links[0])
        if parsed.hostname != 'o.thsi.cn' or not parsed.path.startswith('/ifind-python-aime-tools-service.result-cache/'):
            raise ValueError('Unapproved result-cache host or path')
        url = urlunparse(parsed._replace(scheme='https'))
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if urlparse(response.url).hostname != 'o.thsi.cn' or len(response.content) > 30_000_000:
            raise ValueError('Unexpected CSV redirect/size')
        content = response.content
        frame = None
        for encoding in ['utf-8-sig', 'gb18030']:
            try:
                frame = pd.read_csv(io.BytesIO(content), encoding=encoding)
                break
            except UnicodeError:
                continue
        if frame is None or not len(frame) or len(frame.columns) < 2:
            raise ValueError('Not a usable vendor CSV')
        dest.write_bytes(content)
        item = dict(name=name, downloaded=True, rows=len(frame), columns=frame.columns.tolist(),
            certified_PIT=False, HTTPS_verified=True, automatic_ingestion=False)
        rows.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (OUT / 'download_status.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
