"""Single secure Tushare calendar request using existing local credentials."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_raw/boundary_v66'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import tushare as ts
        pro = ts.pro_api()
        pro._DataApi__http_url = 'https://api.tushare.pro'
        calendar = pro.trade_cal(exchange='SSE', start_date='20050408', end_date='20261014',
            fields='exchange,cal_date,is_open,pretrade_date')
        if calendar is None or not len(calendar) or set(calendar.exchange) != {'SSE'}:
            raise ValueError('Calendar not returned')
        calendar.to_csv(OUT / 'exchange_calendar.csv', index=False)
        status = dict(ok=True, rows=len(calendar), secure_HTTPS=True, attempts=1)
    except Exception as exc:
        status = dict(ok=False, error_type=type(exc).__name__, attempts=1,
            note='No credential-bearing exception text retained')
    (OUT / 'calendar_fetch_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
