"""Bounded iFinD source requests; keep responses quarantined until validated."""
import importlib.util
import json
import os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path('C:/Users/lenovo/.agents/skills/ifind-finance-data/call.py')
OUT = ROOT / 'data_raw/boundary_v66'
QUERIES = [
    ('index', 'index_data', 'dividend_history',
     '沪深300指数000300.SH，2005-04-08至2026-10-14每个月最后一个交易日的历史股息率TTM（%）。'
     '请提供完整历史数据下载CSV，字段含实际证券代码、交易日期、股息率。必须是该历史日期对应的指标，'
     '不得用最新值填历史，不得用红利指数或PE替代。同时说明指标名称、单位、历史成分股口径、'
     'TTM现金分红定义及该指标当日收盘后是否已可见；不能确认的项目请明确标识未知。'),
    ('edb', 'get_edb_data', 'treasury_history',
     '2005-04-08至2026-10-14，中债国债收益率曲线1年和10年期限的历史到期收益率（%），'
     '每个月最后一个交易日，提供完整CSV下载及实际指标代码、观测日期和单位。'
     '只要国债，不要国开债或美债，缺失请保留空值；说明估值日与发布可见日。')
    ,('index', 'index_data', 'equity_close_history',
      '沪深300指数000300.SH，2005-04-08至2026-10-14的每日历史收盘价，'
      '只返回指数实际交易日的日期和收盘价，不要周末节假日，不要把前值填充到非交易日。'
      '请提供完整CSV下载，历史截止日必须为2026-10-14，不使用复权收益或估值字段替代收盘价。')
]


class VerifiedRequests:
    @staticmethod
    def post(*args, **kwargs):
        kwargs['verify'] = True
        return requests.post(*args, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('boundary_ifind', SKILL)
    module = importlib.util.module_from_spec(spec)
    previous = Path.cwd()
    try:
        os.chdir(SKILL.parent)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    module.requests = VerifiedRequests
    log = []
    for server, tool, name, query in QUERIES:
        dest = OUT / (name+'.json')
        if dest.exists():
            print(json.dumps(dict(name=name, cached=True)), flush=True)
            continue
        try:
            response = module.call(server, tool, {'query': query})
            dest.write_text(json.dumps(dict(query=query, response=response,
                certified_PIT=False, automatic_ingestion=False, TLS_verification=True),
                indent=2, ensure_ascii=False), encoding='utf-8')
            item = dict(name=name, ok=bool(response.get('ok')), status_code=response.get('status_code'),
                        requests=1, automatic_ingestion=False)
        except Exception as exc:
            item = dict(name=name, ok=False, error_type=type(exc).__name__, requests=1,
                        note='No credential-bearing exception text logged; no insecure retry')
            log.append(item)
            print(json.dumps(item), flush=True)
            break
        log.append(item)
        print(json.dumps(item), flush=True)
        if not item['ok']:
            break
    (OUT / 'fetch_status.json').write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
