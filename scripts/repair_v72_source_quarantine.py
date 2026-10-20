"""Isolate suspect credit history and rebuild dependent research tensors offline."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v45_data import (local_features, observable_grid, reference_weights,
                         label_reference_policy, annual_splits)
from src.v53_unified_data import independent_cb_relative_labels, validate_relative_dataset
from src.v55_learning_objective import temporal_opportunity_targets, decode_opportunity
from src.v72_source_repair import quarantine_returns, endpoint_reason

REPORT = ROOT / "output/reports/source_quarantine_v72"
DEST = ROOT / "data_processed/source_quarantine_v72"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    inputs = set()

    def read(relative):
        path = ROOT / relative
        inputs.add(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def save(name, obj):
        (REPORT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    source = ROOT / "data_processed/corporate_action_v58/dataset.npz"
    inputs.add(source)
    before_hash = sha(source)
    frozen_status = read("output/reports/corporate_action_v58/status.json")
    if before_hash != frozen_status["dataset_sha256"]:
        raise ValueError("Frozen V58 dataset hash mismatch")
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    original = read("data_processed/corporate_action_v58/research_returns_daily_38.parquet")
    meta = read("output/reports/corporate_action_v58/metadata.json")
    assert len(meta["assets"]) == 38 and meta["assets"] == original.columns.tolist()
    old = read("data_processed/asset_returns_daily.parquet")
    old.date = pd.to_datetime(old.date)
    bad = old.loc[old.asset_code.eq("credit_lgfv_aa") & old.data_source.eq("wind_proxy")].sort_values("date")
    assert bad.date.min() == pd.Timestamp("2018-01-02")
    assert bad.date.max() == pd.Timestamp("2026-05-14")
    # Recorded fingerprints are evidence for quarantine, not a fitted remapping.
    observed = bad.set_index("date").close
    fingerprints = {"2018-01-02": 30515.31, "2018-01-03": 30560.95, "2018-01-11": 31120.39}
    for day, expected in fingerprints.items():
        assert np.isclose(observed.loc[day], expected, atol=.005, rtol=0)
    read("data_raw/wind/index_kline_H11098.CSI_20180101_20260602.json")
    read("data_raw/refresh_20261014_wind_verified/wind/credit_lgfv_aa_wind_sh_identity.json")
    returns = quarantine_returns(original, "credit_lgfv_aa", "2026-05-15")
    incident = dict(asset="credit_lgfv_aa", issue="historical_source_identity_mismatch_suspected_Hang_Seng",
        bad_source="wind_proxy", source_rows=len(bad), source_start="2018-01-02", source_end="2026-05-14",
        quarantine_return_through="2026-05-15", method="mask_source_segment_and_boundary_no_rescaling",
        fingerprints=fingerprints, original_artifacts_preserved=True,
        correct_full_history_required=True, resumed_segment_status="existing_ifind_Wind_overlap_evidence_not_full_total_return_certification",
        corroborating_primary_urls=[
            "https://www1.hkexnews.hk/listedco/listconews/sehk/2019/0110/ltn20190110625.pdf",
            "https://www.sse.com.cn/market/sseindex/indexlist/indexdetails/indexmethods/c1/H11098_h11098hbook_CN.pdf"],
        additional_dated_corroboration=[
            "https://www.citics.com.hk/upload/files/report/20180108025908_1337.pdf",
            "https://www.cinda.com.hk/attachment/research_consultancy/pdf/1515721058q8DW.pdf"],
        evidence_limit="Price fingerprints plus incompatible later bond identity; not a supplier-certified full HSI identity history")
    save("identity_incident.json", incident)

    # Diagnose existing gaps without hindsight fill, stale-limit relaxation or synthetic trades.
    rows = []
    for asset in ("equity_cyclical", "equity_chinext", "credit_lgfv_aa"):
        levels = old.loc[old.asset_code.eq(asset)].set_index("date").close.reindex(returns.index)
        valid = original[asset].dropna()
        ix = original.index[(original.index >= valid.index[0]) & (original.index <= valid.index[-1]) & original[asset].isna()]
        for day in ix:
            i = returns.index.get_loc(day)
            reason = endpoint_reason(levels.iloc[i - 1], levels.iloc[i])
            if asset == "credit_lgfv_aa" and day <= pd.Timestamp("2026-05-15"):
                reason = "identity_quarantine_supersedes_gap_repair"
            rows.append(dict(asset=asset, return_date=day, source_reason=reason,
                             action="retain_missing_until_source_and_execution_evidence"))
    reits = read("data_processed/unified_v53/reits_nine_component_unadjusted_close.parquet")
    s = original.alt_reits
    for day in s.index[s.isna() & (s.index >= s.first_valid_index()) & (s.index <= s.last_valid_index())]:
        i = reits.index.get_loc(day)
        missing = reits.columns[reits.iloc[i].isna() | reits.iloc[i - 1].isna()].tolist()
        rows.append(dict(asset="alt_reits", return_date=day, source_reason="component_endpoint_missing:" + ";".join(missing),
                         action="do_not_renormalize_surviving_components"))
    us = read("data_processed/unified_v53/us_curve_visibility_audit.parquet")
    fx = read("data_processed/unified_v53/cny_parity_visibility_audit.parquet")
    s = original.alt_us_bond
    for day in s.index[s.isna() & (s.index >= s.first_valid_index()) & (s.index <= s.last_valid_index())]:
        i = s.index.get_loc(day)
        missing = []
        for label, series in (("US_yield", us.yield_percent), ("FX_parity", fx.cny_per_usd)):
            if not np.isfinite(series.iloc[i]) or not np.isfinite(series.iloc[i - 1]):
                missing.append(label)
        rows.append(dict(asset="alt_us_bond", return_date=day, source_reason="visibility_endpoint_missing:" + ";".join(missing),
                         action="preserve_7_calendar_day_tolerance_no_zero_fill"))
    pd.DataFrame(rows).to_csv(REPORT / "non_cb_gap_causes.csv", index=False, encoding="utf-8-sig")
    missing = read("output/reports/asset_dependency_v71/cb_selected_missing_observations.csv")
    basic = read("data_raw/cb_database_20261014_v49/basic.parquet").set_index("ts_code")
    close = read("data_processed/cb_evidence_v50/quotes/close_individual.parquet")
    volume = read("data_processed/cb_evidence_v50/quotes/volume_individual.parquet")
    preclose = read("data_processed/cb_evidence_v50/quotes/preclose_individual.parquet")
    raw_returns = read("data_processed/cb_evidence_v50/individual_observed_price_returns.parquet")
    cb_rows = []
    for (code, date), group in missing.groupby(["security", "return_date"]):
        day = pd.Timestamp(date)
        ix = close.index.get_loc(day)
        prior, current, vol, pc = (close.at[close.index[ix - 1], code], close.at[day, code],
                                   volume.at[day, code], preclose.at[day, code])
        reason = endpoint_reason(prior, current)
        if np.isfinite(current) and (not np.isfinite(vol) or vol <= 0):
            reason += ";no_positive_volume_transaction"
        record = dict(security=code, return_date=day, affected_buckets=len(group),
            prior_close=prior, current_close=current, source_preclose=pc, source_volume=vol,
            current_research_return=raw_returns.at[day, code], source_reason=reason,
            repair_status="not_repaired_no_cashflow_or_execution_inference")
        if code in basic.index:
            for field in ("bond_short_name", "delist_date", "maturity_date"):
                record["current_registry_" + field] = basic.at[code, field]
        record["registry_use"] = "diagnostic_only_never_historical_eligibility"
        cb_rows.append(record)
    pd.DataFrame(cb_rows).to_csv(REPORT / "cb_27_security_days_root_causes.csv", index=False, encoding="utf-8-sig")
    print("V72 rebuilding local features and nine-grid", flush=True)
    bars = {k: read(f"data_processed/unified_v53/source_{k}.parquet") for k in ("open", "high", "low", "close", "vol", "amount")}
    arrays["returns"] = returns.fillna(0).to_numpy(dtype=np.float32)
    arrays["return_masks"] = returns.notna().to_numpy()
    arrays["local_values"], arrays["local_masks"], feature_names = local_features(returns, bars)
    arrays["macro_values"], arrays["macro_masks"], arrays["regime_index"] = observable_grid(returns)
    preferred, policy = label_reference_policy(meta["assets"], meta["asset_modules"])
    sw = reference_weights(arrays["returns"], arrays["return_masks"], preferred, arrays["module_index"])
    arrays["sleeve_reference_weights"] = sw
    cb_ref = read("data_processed/unified_v53/cb_reference_returns.parquet").iloc[:, 0].to_numpy()
    arrays.update(independent_cb_relative_labels(arrays["dates"], arrays["returns"], arrays["return_masks"],
        arrays["module_index"], sw, arrays["balanced_reference_weights"], cb_ref))
    sr = np.sum(sw * arrays["returns"][:, None, :], axis=-1)
    sm = (sw.sum(-1) > 0) & ((sw == 0) | arrays["return_masks"][:, None, :]).all(-1)
    sr[:, 2], sm[:, 2] = cb_ref, np.isfinite(cb_ref)
    for task, daily in (("asset", returns.to_numpy()), ("sleeve", np.where(sm, sr, np.nan))):
        y, mask = arrays[task + "_target"].copy(), arrays[task + "_target_mask"].copy()
        result = temporal_opportunity_targets(y, mask, daily)
        arrays[task + "_raw_relative_target"], arrays[task + "_raw_relative_target_mask"] = y, mask
        for suffix, key in (("_target", "target"), ("_target_mask", "mask"),
                            ("_opportunity_scale", "scale"), ("_opportunity_offset", "offset")):
            arrays[task + suffix] = result[key]
        good = result["mask"]
        assert np.allclose(decode_opportunity(result["target"], result["scale"], result["offset"])[good], y[good], atol=1e-6)
    validate_relative_dataset(arrays)
    a = meta["assets"].index("credit_lgfv_aa")
    quarantined = returns.index <= pd.Timestamp("2026-05-15")
    assert not arrays["return_masks"][quarantined, a].any()
    assert not arrays["local_masks"][quarantined, a].any()
    assert not arrays["asset_raw_relative_target_mask"][quarantined, :, a].any()
    with np.load(source, allow_pickle=False) as parent:
        # Nine-grid relies on rates/equity, not the quarantined credit token.
        assert np.array_equal(arrays["regime_index"], parent["regime_index"])
        changes = []
        for j, asset in enumerate(meta["assets"]):
            changes.append(dict(asset=asset, old_valid_returns=int(parent["return_masks"][:, j].sum()),
                new_valid_returns=int(arrays["return_masks"][:, j].sum()),
                old_local_feature_cells=int(parent["local_masks"][:, j].sum()),
                new_local_feature_cells=int(arrays["local_masks"][:, j].sum()),
                old_opportunity_label_cells=int(parent["asset_target_mask"][:, :, j].sum()),
                new_opportunity_label_cells=int(arrays["asset_target_mask"][:, :, j].sum())))
    pd.DataFrame(changes).to_csv(REPORT / "38_asset_impact.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(DEST / "dataset.npz", **arrays)
    returns.to_parquet(DEST / "research_returns_daily_38.parquet")
    splits = []
    for year in range(2022, 2027):
        split = annual_splits(arrays, year)
        splits.append(dict(year=year, reason=split["reason"], **{k: len(split[k]) for k in ("train", "validation", "gate", "test")}))
    pd.DataFrame(splits).to_csv(REPORT / "annual_split_readiness.csv", index=False)
    metadata = copy.deepcopy(meta)
    metadata.update(version="v72_identity_quarantined_research_not_execution_certified",
        parent_dataset_sha256=before_hash, dataset_sha256=sha(DEST / "dataset.npz"), local_features=feature_names,
        source_identity_quarantine=incident, old_checkpoints_incompatible=True,
        balanced_reference_anchor_lineage_certified=False,
        source_manifest={str(p.relative_to(ROOT)): dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sorted(inputs)},
        array_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()})
    metadata.setdefault("limitations", []).append("credit_lgfv_aa pre-2026-05-18 history quarantined for incompatible source identity; full correct history pending")
    save("metadata.json", metadata)
    assert sha(source) == before_hash
    status = dict(status="rebuilt_identity_quarantine_not_trained", assets=38,
        dataset_sha256=sha(DEST / "dataset.npz"), parent_dataset_sha256=before_hash,
        newly_masked_return_days=int(original.credit_lgfv_aa.notna().sum() - returns.credit_lgfv_aa.notna().sum()),
        remaining_credit_aa_observed_returns=int(returns.credit_lgfv_aa.notna().sum()),
        credit_aa_opportunity_labels=int(arrays["asset_target_mask"][:, :, a].sum()),
        nine_grid_preserved_and_unchanged=True, original_dataset_unchanged=True,
        models_retrained=False, ppo_started=False, formal_promotion=False, all_cashflows_certified=False,
        formal_training_allowed=False, balanced_reference_anchor_lineage_certified=False,
        full_correct_credit_history_pending=True, market_data_downloaded=False,
        current_registry_used_for_diagnosis_only=True)
    save("status.json", status)
    save("source_hashes.json", {str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)} |
         {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)), "src/v72_source_repair.py": sha(ROOT / "src/v72_source_repair.py")})
    report = f"""# V72 来源隔离修复与剩余阻塞

## 重大更正

V71只验证了来源登记、收益重建及掩码一致，不能证明真实证券身份正确。本轮追溯发现 credit_lgfv_aa 的 wind_proxy 旧段与后续沪城投债身份不相容，不能仅屏蔽接缝的一天。

旧段2018-01-02为30515.31、2018-01-03为30560.95、2018-01-11为31120.39，与恒生指数历史点位吻合；当前注册身份是 H11098.SH 上证城投债。香港交易所披露材料支持首个点位核验，上交所编制方案支持债券指数身份。完整证据链接见 identity_incident.json。我们据此将旧段定为不可用于信用训练；没有把统计特征当作重新标注或换基的依据。

## 已执行的修复

- 单独生成 data_processed/source_quarantine_v72/dataset.npz，保留38个资产槽位与九宫格。
- 隔离旧来源及2026-05-15接缝收益；新增屏蔽 {status['newly_masked_return_days']} 个原有效收益。当前后段仅余 {status['remaining_credit_aa_observed_returns']} 个有效日，不冒充2018全历史。
- 重建局部特征、九宫格、参考权重、原始相对标签和机会目标/反变换参数。九宫格只依赖原利率/权益证据，因此与V58完全一致。
- 该资产机会目标有效单元为 {status['credit_aa_opportunity_labels']}，不够成熟历史就保持不可训练，不借用被隔离的旧统计。
- 不覆盖V58、不修改模型或配置，不沿用旧checkpoint，不启动PPO或发布新净值。
- 原 balanced_reference_weights 沿用V58的既定参考，避免擅自改变学习任务；其上游V44锚点是否也使用过错误来源尚未认证。因此V72仍关闭正式训练许可，不能把“重新计算了标签”理解为整条参考链已无污染。

## 缺口根因清单

non_cb_gap_causes.csv 将城投旧段缺口归入身份隔离；周期ETF/创业板ETF按前后报价端点定位；REITs逐项列出缺报价成分；美债按美国利率和FX可见端点分开定位。未擅自填补、用幸存成分归一或放宽7天可见期限。

cb_27_security_days_root_causes.csv 给出22只券、27个证券日的收盘、前收、成交量、受影响篮子数和当前登记表中的退出/到期线索。当前登记表只用于事后定位，未倒灌当时可买名单；已有报价/无成交也不等于可以卖出。此表不是27条已认证赎回事件。

## 旧结果如何处理

依赖这段数据的旧训练与历史回测，不能继续标为通过真实来源审计。它影响信用资产的预测及注意力上下文，不能通过从旧回测中简单减一列收益来修正。保留旧结果供审计追溯，后续须用正确来源重建并重训验证。V72是隔离版，不是已补齐的正式替代。

## 下一步

优先重新取得 H11098.SH 全历史，并认证逐段证券身份和收益口径，再替换隔离槽位；同步核验7个ETF分红/拆分、9个REIT分派和实际转债成员现金流。未完成前不能声称无严重数据问题，也不能将新的训练损失或夏普当作修复成功证据。

运行：python -X utf8 scripts/repair_v72_source_quarantine.py。38_asset_impact.csv 为逐资产影响，annual_split_readiness.csv 为既定purge划分检查。2024—2026已多次观察，不是全新样本外。
"""
    (REPORT / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
