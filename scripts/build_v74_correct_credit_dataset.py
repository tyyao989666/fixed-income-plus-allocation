"""Apply V73 verified index history to an isolated V72-derived research pack."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v74_rebuild import exact_calendar_returns, rebuild
from src.v45_data import annual_splits

OUT = ROOT / "output/reports/correct_credit_v74"
DEST = ROOT / "data_processed/correct_credit_v74"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    inputs = {ROOT / "src/v74_rebuild.py", ROOT / "src/v45_data.py",
              ROOT / "src/v53_unified_data.py", ROOT / "src/v55_learning_objective.py",
              Path(__file__)}

    def read(relative):
        path = ROOT / relative
        inputs.add(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text(encoding="utf-8"))

    def save(name, value):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    status = read("output/reports/credit_identity_v73/status.json")
    assert status["status"] == "exact_history_candidate_downloaded_not_activated" and status["exact_identity_verified"]
    raw_hashes = read("output/reports/credit_identity_v73/raw_hashes.json")
    for name, expected in raw_hashes.items():
        path = ROOT / name
        inputs.add(path)
        assert sha(path) == expected, name
    meta = read("output/reports/source_quarantine_v72/metadata.json")
    parent_status = read("output/reports/source_quarantine_v72/status.json")
    parent = ROOT / "data_processed/source_quarantine_v72/dataset.npz"
    inputs.add(parent)
    assert sha(parent) == parent_status["dataset_sha256"]
    with np.load(parent, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    returns = read("data_processed/source_quarantine_v72/research_returns_daily_38.parquet")
    levels = read("data_raw/credit_identity_v73/correct_credit_index_levels.parquet")
    fresh = exact_calendar_returns(levels, returns.index, "H11098.SH")
    asset = "credit_lgfv_aa"
    paired = pd.concat([returns[asset].rename("existing"), fresh.rename("correct")], axis=1).dropna()
    assert len(paired) >= 90
    assert np.allclose(paired.existing, paired.correct, atol=5e-6, rtol=.001)
    before = read("data_processed/corporate_action_v58/research_returns_daily_38.parquet")
    restored = int((returns[asset].isna() & fresh.notna()).sum())
    returns[asset] = fresh
    other = [a for a in returns if a != asset]
    assert np.allclose(returns[other], before[other], equal_nan=True)
    bars = {k: read(f"data_processed/unified_v53/source_{k}.parquet") for k in ("open", "high", "low", "close", "vol", "amount")}
    cb = read("data_processed/unified_v53/cb_reference_returns.parquet").iloc[:, 0].to_numpy()
    print("V74 rebuilding 38-asset features, nine-grid and all causal labels", flush=True)
    result, names = rebuild(arrays, returns, bars, meta, cb)
    assert len(meta["assets"]) == 38
    assert np.array_equal(result["regime_index"], arrays["regime_index"])
    for h, j in ((5, 0), (20, 1), (60, 2)):
        assert not result["asset_raw_relative_target_mask"][-h:, j].any()
    np.savez_compressed(DEST / "dataset.npz", **result)
    returns.to_parquet(DEST / "research_returns_daily_38.parquet")
    a = meta["assets"].index(asset)
    comparisons = []
    for j, code in enumerate(meta["assets"]):
        comparisons.append(dict(asset=code, valid_returns=int(result["return_masks"][:, j].sum()),
            missing_returns=int((~result["return_masks"][:, j]).sum()),
            local_feature_cells=int(result["local_masks"][:, j].sum()),
            raw_label_cells=int(result["asset_raw_relative_target_mask"][:, :, j].sum()),
            opportunity_label_cells=int(result["asset_target_mask"][:, :, j].sum())))
    pd.DataFrame(comparisons).to_csv(OUT / "38_asset_coverage.csv", index=False, encoding="utf-8-sig")
    contrast = pd.DataFrame(dict(old_v58_return=before[asset], correct_v74_return=fresh))
    contrast.to_csv(OUT / "credit_old_vs_correct_returns.csv", encoding="utf-8-sig")
    splits = []
    for year in range(2022, 2027):
        split = annual_splits(result, year)
        splits.append(dict(year=year, reason=split["reason"], **{k: len(split[k]) for k in ("train", "validation", "gate", "test")}))
    pd.DataFrame(splits).to_csv(OUT / "annual_split_readiness.csv", index=False)
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)}
    metadata = copy.deepcopy(meta)
    metadata.update(version="v74_correct_credit_index_identity_research_not_execution_certified",
        dataset_sha256=sha(DEST / "dataset.npz"), parent_dataset_sha256=sha(parent),
        source_identity_quarantine=dict(status="historical_mismatch_replaced_in_this_pack_only", asset=asset,
            source_identifier="H11098.SH", price_history_complete=True, strict_AA_universe=False),
        asset_semantic_overrides={asset: "上证城投债指数代理（非严格AA评级券池）"},
        local_features=names, old_checkpoints_incompatible=True,
        source_manifest={name: dict(path=name, sha256=value) for name, value in manifest.items()},
        array_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in result.items()})
    metadata["limitations"] = [x for x in metadata.get("limitations", []) if "pre-2026-05-18 history quarantined" not in x]
    metadata["limitations"].append("V74 correct source identity does not certify legacy V44 balanced reference anchor, ETF/bucket cashflows or executability")
    metadata["asset_names"][a] = "上证城投债代理（非严格AA）"
    save("metadata.json", metadata)
    final = dict(status="correct_credit_history_rebuilt_not_trained", assets=38, price_rows=len(levels),
        start=str(returns.index.min().date()), end=str(returns.index.max().date()),
        credit_valid_returns=int(fresh.notna().sum()), credit_internal_missing_returns=int(fresh.iloc[1:].isna().sum()),
        restored_vs_v72=restored, raw_relative_label_cells=int(result["asset_raw_relative_target_mask"][:, :, a].sum()),
        opportunity_label_cells=int(result["asset_target_mask"][:, :, a].sum()),
        max_abs_credit_daily_return=float(fresh.abs().max()), recent_return_overlap_rows=len(paired),
        other_37_returns_unchanged=True, nine_grid_unchanged=True, checkpoints_must_be_retrained=True,
        formal_training_allowed=False, balanced_reference_anchor_lineage_certified=False,
        coupon_and_terminal_cashflows_certified=False, formal_promotion=False, ppo_started=False,
        dataset_sha256=sha(DEST / "dataset.npz"), source="Wind financial data service")
    save("status.json", final)
    save("source_hashes.json", manifest)
    assert all(sha(ROOT / name) == value for name, value in manifest.items())
    text = f"""# V74 城投债历史来源修复

## 本轮实际完成

通过精确代码 H11098.SH 从万得重新取得2018-01-02至2026-10-14的逐年历史，共{len(levels)}行。先核实上证城投债名称、人民币和代码，再用2018短区间、15个近期重合日与逐年响应交叉核验。拒绝非中国时区、空价格、重复日期、区间截断和历史/探针不一致，不缩放旧价格、不补零。

重新生成 data_processed/correct_credit_v74/dataset.npz。38资产和九宫格保留，其他37资产收益不改；修复资产有{final['credit_valid_returns']}个有效日收益、内部缺口{final['credit_internal_missing_returns']}日。相对V72恢复{restored}个日收益，同时把V58旧段的错误收益替换为正确来源。

已重建特征、参考权重、原始5/20/60日相对收益、机会目标及反变换参数。该资产原始标签{final['raw_relative_label_cells']}格，机会标签{final['opportunity_label_cells']}格；年度训练/验证/门控/测试划分仍用既定purge和成熟截止，无按测试表现选参数。

## 经济口径

这个槽位沿用 credit_lgfv_aa 只是为了兼容38资产ID，显示名称改为“上证城投债代理（非严格AA）”。上交所编制方案规定投资级以上、剩余期限1年以上城投债，不是AA专属篮子。方案使用估值全价与派息计算；本轮使用供应商指数点位变化，不再另外加一次票息，更不把估值指数称为实际成交组合。

官方定义：[上交所编制方案](https://www.sse.com.cn/market/sseindex/indexlist/indexdetails/indexmethods/c1/H11098_h11098hbook_CN.pdf)。现行方案不自动认证所有历史版本或交易成本。

## 仍未完成，不得误报

V44 balanced_reference_weights 锚点的上游污染尚需核查。7个ETF现金流、9个REIT到账及转债实际持仓的付息/退出/冻结库存也未全部认证。V74是新研究包，没有自动切换旧流水线、没有重训、没有PPO或新夏普；正式训练与晋升仍关闭。源身份修复不等于整体可执行收益已认证。

## 复核

下载：python -X utf8 scripts/acquire_v73_correct_credit_history.py（成功缓存复用，不重复消耗请求）。

重建：python -X utf8 scripts/build_v74_correct_credit_dataset.py。

38_asset_coverage.csv、credit_old_vs_correct_returns.csv 和 annual_split_readiness.csv 是覆盖、旧新差异与划分底稿；source_hashes.json 和V73原始响应提供可追溯证据。旧V58/V72和模型未覆盖。

数据来源于万得 Wind 金融数据服务。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
