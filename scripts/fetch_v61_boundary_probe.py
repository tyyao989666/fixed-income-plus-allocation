"""One bounded iFinD schema probe; no credential logging or automatic retries."""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("C:/Users/lenovo/.agents/skills/ifind-finance-data/call.py")


def main():
    out = ROOT / "data_raw/state_momentum_v61"
    out.mkdir(parents=True, exist_ok=True)
    query = "请提供沪深300指数000300.SH在2015年每个月最后交易日的历史股息率(TTM，百分比)和交易日期，12个月共12行。不要用中证红利指数价格或PE替代。请返回实际历史指标名称、单位、数据来源，不支持请明确说明。"
    request = dict(server="index", tool="index_data", query=query)
    try:
        spec = importlib.util.spec_from_file_location("ifind_boundary_probe", SKILL)
        module = importlib.util.module_from_spec(spec)
        previous = Path.cwd()
        try:
            os.chdir(SKILL.parent)
            spec.loader.exec_module(module)
        finally:
            os.chdir(previous)
        response = module.call("index", "index_data", {"query": query})
        result = dict(request=request, response=response, certified_PIT=False,
                      note="Schema probe only; no automatic ingestion or extrapolation")
        (out / "ifind_dividend_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(saved=True, ok=response.get("ok"), status_code=response.get("status_code")), ensure_ascii=False))
    except Exception as exc:
        # Exception strings may contain request URLs; do not persist credentials.
        (out / "ifind_dividend_probe_status.json").write_text(json.dumps(dict(request=request,
            status="failed", error_type=type(exc).__name__, attempts=1, certified_PIT=False), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, attempts=1)))


if __name__ == "__main__":
    main()
