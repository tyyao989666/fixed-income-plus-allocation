"""Bounded iFinD samples for retired CBs. Never auto-promote returned tables."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw/off_exchange_quotes_v91"
OUT = ROOT / "output/reports/off_exchange_quotes_v91"
CLI = Path.home() / ".codex/skills/ifind-finance-data/call-node.js"
JS = "const {call}=require(process.argv[1]);call('bond','bond_market_data',JSON.parse(process.argv[2])).then(x=>process.stdout.write(JSON.stringify(x))).catch(e=>{process.stderr.write(e.name||'Error');process.exitCode=1});"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("404001", "蓝盾退债", "2023-09-18", "2023-09-22"),
        ("404002", "搜特退债", "2023-10-20", "2023-10-27"),
        ("404004", "汇车退债", "2024-11-01", "2024-11-08"),
    ]
    rows = []
    for code, name, start, end in tasks:
        query = (f"退市板块可转换债券{name}，债券代码{code}，非同名股票，{start}至{end}，"
                 "逐日原始收盘成交价、成交量及其单位。保留证券代码和交易日期。不要复权、向前填充、参考前收盘价或估算值。"
                 "无成交时保留缺失，不要用股票价格代替。")
        path = RAW / f"{code}_{start}_{end}.json"
        began = time.monotonic()
        if path.exists():
            receipt = json.loads(path.read_text(encoding="utf-8"))
        else:
            receipt = dict(query=query, code=code, start=start, end=end,
                           fetched_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           provider="ifind_bond_market_data", approved_for_returns=False)
            try:
                proc = subprocess.run(["node", "-e", JS, str(CLI), json.dumps(dict(query=query), ensure_ascii=False)],
                                      capture_output=True, encoding="utf-8", errors="replace", timeout=100)
                receipt["exit_code"] = proc.returncode
                if proc.stdout.strip():
                    receipt["response"] = json.loads(proc.stdout)
                    receipt["status"] = "captured_requires_semantic_review"
                else:
                    receipt["status"] = "provider_call_failed"
                    receipt["error_type"] = proc.stderr.strip()[:80]
            except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                receipt.update(status="provider_call_failed", error_type=type(exc).__name__)
            receipt["elapsed_seconds"] = round(time.monotonic() - began, 2)
            path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(dict(code=code, status=receipt["status"], seconds=receipt.get("elapsed_seconds"),
                         raw_file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         approved_for_returns=False))
        (OUT / "probe_status.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        response_text = json.dumps(receipt.get("response", {}), ensure_ascii=False).lower()
        if receipt["status"] == "provider_call_failed" or any(t in response_text for t in ["unauthorized", "quota exceeded", "invalid token", '"status_code": 401', '"status_code": 403']):
            print("Stopped bounded provider probe after service/auth failure; remaining tasks not requested", flush=True)
            break


if __name__ == "__main__":
    main()
