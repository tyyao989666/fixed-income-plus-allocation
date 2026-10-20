"""Audit acquired files without promoting candidate inputs or fitting models."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage_20261014_market_data import DEST, END, tables, save


def main():
    manifest = json.loads((DEST / "manifest.json").read_text(encoding="utf-8"))
    checks = []
    for name, digest in manifest["files"].items():
        path = DEST / name
        checks.append(dict(check="sha256:"+name, passed=path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest))
    calendar = pd.read_parquet(DEST / "tushare_cb/calendar.parquet")
    sessions = set(calendar.cal_date.astype(str))
    days, bonds = [], set()
    records = 0
    for path in sorted((DEST / "tushare_cb").glob("cb_daily_*.parquet")):
        frame = pd.read_parquet(path)
        date = path.stem.split("_")[-1]
        days.append(date)
        records += len(frame)
        bonds.update(frame.ts_code.astype(str))
        checks.append(dict(check="CB_identity_date:"+date, passed=bool(
            not frame.empty and not frame.ts_code.duplicated().any() and set(frame.trade_date.astype(str)) == {date})))
    checks.append(dict(check="CB_complete_calendar", passed=set(days) == sessions and max(days) == END))
    status = pd.read_csv(DEST / "coverage_status.csv")
    valid = status.loc[status.status.eq("validated_candidate")]
    checks.append(dict(check="not_promoted", passed=manifest["training_ready"] is False and manifest["formal_replacement"] is False))
    macro = []
    for path in sorted((DEST / "wind").glob("macro_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for frame, _ in tables(record["response"]):
            if "period_date" not in frame:
                continue
            dates = pd.to_datetime(frame.period_date.astype(str), errors="coerce")
            for name, group in frame.groupby("name", dropna=False):
                macro.append(dict(request=path.stem, indicator=name, source=str(group.source.iloc[0]),
                                  unit=str(group.unit.iloc[0]), rows=len(group),
                                  nonmissing=int(pd.to_numeric(group.value, errors="coerce").notna().sum()),
                                  actual_release_date_available=False))
            checks.append(dict(check="macro_period_cutoff:"+path.stem, passed=bool(dates.notna().all() and dates.le(pd.Timestamp(END)).all())))
    pd.DataFrame(macro).to_csv(DEST / "macro_inventory.csv", index=False, encoding="utf-8-sig")
    passed = all(c["passed"] for c in checks)
    save(DEST / "acquisition_audit.json", dict(passed=passed, checks=checks, training_ready=False,
        CB_days=len(days), CB_rows=records, CB_unique_bonds=len(bonds), validated_asset_candidates=len(valid),
        source_PIT_certified=False, caveat="Integrity is not economic-definition or release-time certification"))
    lines = ["# 截至 2026-10-14 的数据抓取验收", "",
             f"文件与日期完整性检查：{sum(c['passed'] for c in checks)}/{len(checks)}。原数据和模型未覆盖；尚未达到训练就绪条件。", "",
             f"- 24 条原有资产序列候选中，通过身份及重叠点位检查：{len(valid)} 条。",
             "- 另取得 9 只既有 REITs 成分行情和 3 条辅助指数行情；尚未重新构造收益或状态特征。",
             f"- 转债原始日行情：{len(days)} 个交易日、{records:,} 条记录、{len(bonds)} 个代码，末日 {max(days)}。",
             "- 相对 2026-07-15 正式冻结点新增 59 个交易日，不等于增加 59×38 个独立样本。", "",
             "## 仍不能训练的原因", "",
             "- 转债历史余额、新发/退市与条款的真实可见日未同步完备。当前基本资料是抓取日快照，不能回填历史；12 风格桶尚未重建。",
             "- REITs 的分红、收益口径与合成权重未完成核对；美债仍需确定同口径收益率、汇率与国内可见日。",
             "- 信用评级/产业城投代理标签不能按字面解释；30Y 仍是 10Y 以上国债财富指数代理。",
             "- Wind EDB 的多指标自然语言查询可能只返回一部分指标。取到了原始响应不代表全部国债期限、信用曲线及资金指标齐全。",
             "- 月度 EDB 响应目前没有通过真实发布日期与历史版本验收，不进入新的 PIT 状态卡；未发布值不填。",
             "- 成交量/额单位、ETF 复权和跨源特征拼接仍需验收。债券指数 OHLC 不能作为真实债券日内行情。", "",
             "## 下一步顺序", "",
             "补齐上述缺口并重建 38 资产 PIT 特征后，先用冻结的 iTransformer/PPO 检验新增时段；不调整旧模型参数去贴合新时段。随后才为新的发行期做因果重训，5/20/60 日标签分别按成熟情况掩码。", "",
             "方案：`docs/data_refresh_and_retraining_decision_20261015.md`。本报告是抓取验收，不是策略效果报告或完整 PIT 认证。"]
    (DEST / "acquisition_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(passed=passed, checks=len(checks), validated_assets=len(valid),
                         cb_days=len(days), cb_rows=records, cb_codes=len(bonds), macro_series=len(macro)), ensure_ascii=False))
    if not passed:
        raise ValueError("Acquisition integrity audit failed")


if __name__ == "__main__":
    main()
