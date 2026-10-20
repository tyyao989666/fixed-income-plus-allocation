"""One desktop Wind session probe, no UI/login prompt and no repeated connection."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_raw/boundary_v66'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from WindPy import w
        result = w.start(waitTime=10, showmenu=False)
        status = dict(connected=bool(w.isconnected()), error_code=getattr(result, 'ErrorCode', None),
            prompts_disabled=True, attempts=1, note='SDK probe only; field semantics not inferred')
    except Exception as exc:
        status = dict(connected=False, error_type=type(exc).__name__, attempts=1, prompts_disabled=True)
    (OUT / 'wind_sdk_probe.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
