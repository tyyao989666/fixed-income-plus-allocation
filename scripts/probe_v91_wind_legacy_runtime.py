"""Run with -X utf8=0: old WindPy opens a legacy-encoded .pth file.

Do not modify the SDK, .pth, registry, environment or the user's Wind session.
"""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    folder = ROOT / "data_raw/off_exchange_quotes_v91"
    path = folder / "wind_legacy_runtime_probe.json"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
        return
    record = dict(utf8_mode=sys.flags.utf8_mode, runtime_only_encoding_change=True,
                  observed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  prompts_disabled=True, connection_attempts=0, quote_requests=0)
    try:
        if sys.flags.utf8_mode:
            raise ValueError("Use -X utf8=0 for this isolated compatibility probe")
        from WindPy import w
        record["sdk_imported"] = True
        record["connection_attempts"] = 1
        result = w.start(waitTime=10, showmenu=False)
        record.update(connected=bool(w.isconnected()), error_code=getattr(result, "ErrorCode", None))
        if record["connected"]:
            # One small, unfilled historical sample. Identity and volume need audit.
            record["quote_requests"] = 1
            quote = w.wsd("404001.NQ", "close,volume", "2023-09-18", "2023-09-22", "Fill=Blank")
            sample = dict(code="404001.NQ", fields=getattr(quote, "Fields", None),
                          codes=getattr(quote, "Codes", None), data=getattr(quote, "Data", None),
                          times=[str(x) for x in getattr(quote, "Times", [])],
                          error_code=getattr(quote, "ErrorCode", None), approved_for_returns=False)
            (folder / "wind_404001_sample.json").write_text(json.dumps(sample, default=str, indent=2), encoding="utf-8")
    except Exception as exc:
        record.update(connected=False, error_type=type(exc).__name__)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
