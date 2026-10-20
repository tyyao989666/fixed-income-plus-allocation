"""One local desktop SDK connection probe, no login UI or repeated launch."""
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "data_raw/off_exchange_quotes_v91"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "wind_sdk_probe.json"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
        return
    record = dict(attempts=1, prompts_disabled=True, quote_requests=0,
                  observed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        from WindPy import w
        result = w.start(waitTime=10, showmenu=False)
        record.update(connected=bool(w.isconnected()), error_code=getattr(result, "ErrorCode", None))
    except Exception as exc:
        record.update(connected=False, error_type=type(exc).__name__)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
