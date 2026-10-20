"""Read-only source/dependency audit of the V58 38-asset research dataset."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v49_point_in_time import cb_bucket_weights, realized_bucket_return

OUT = ROOT / "output/reports/asset_dependency_v71"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def coverage(series):
    valid = np.isfinite(series) & series.gt(-1)
    positions = np.flatnonzero(valid)
    internal = int((~valid.iloc[positions[0]:positions[-1] + 1]).sum()) if len(positions) else 0
    return dict(valid_days=int(valid.sum()), missing_days=int((~valid).sum()),
                missing_ratio=float((~valid).mean()), internal_missing_days=internal,
                first_valid=str(series.index[positions[0]].date()) if len(positions) else "",
                last_valid=str(series.index[positions[-1]].date()) if len(positions) else "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = set()
    lineage_path = ROOT / "output/reports/cb_masked_research_v51/status.json"
    inputs.add(lineage_path)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    for relative, expected in lineage["source_sha256"].items():
        path = ROOT / relative
        inputs.add(path)
        if digest(path) != expected:
            raise ValueError(f"V51 frozen input changed: {relative}")

    def read(relative):
        path = ROOT / relative
        inputs.add(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    meta_path = ROOT / "output/reports/corporate_action_v58/metadata.json"
    inputs.add(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assets = meta["assets"]
    names = dict(zip(assets, meta["asset_names"]))
    returns = read("data_processed/corporate_action_v58/research_returns_daily_38.parquet")
    npz_path = ROOT / "data_processed/corporate_action_v58/dataset.npz"
    inputs.add(npz_path)
    with np.load(npz_path) as z:
        assert len(assets) == 38 and list(returns.columns) == assets
        assert np.array_equal(pd.to_datetime(z["dates"]), returns.index)
        valid = np.isfinite(returns.to_numpy()) & (returns.to_numpy() > -1)
        assert np.array_equal(valid, z["return_masks"])
        assert np.allclose(returns.to_numpy()[valid], z["returns"][valid], atol=1e-7, rtol=1e-6)
        assert np.all(z["returns"][~valid] == 0)
        label_masks = z["asset_raw_relative_target_mask"].copy()
    contracts = read("output/reports/unified_v53/asset_contracts.csv").set_index("asset_code")
    source_ids = read("data_raw/refresh_20261014_wind_verified/coverage_status.csv").set_index("dataset")
    inventory = []
    for a in assets:
        if a.startswith("cb_"):
            kind, source = "self_built_CB", "V51 prior-close constituent basket"
            basis = "Price-only daily reconstitution; balance weights capped at 10%; no certified coupon/terminal cash"
            action = "P0: certify actual held cashflows/exits; preserve missing masks; disclose fallback and price-based style"
        elif a == "alt_reits":
            kind, source = "self_built_REIT", "9 fixed REIT components"
            basis = "Equal-weight all-9-complete ex-date distribution research return; not payment-date cash ledger"
            action = "P0: validate 9 component distributions/payment dates; no invented pre-listing history"
        elif a in ("alt_us_bond", "rate_30y_curve_proxy"):
            kind, source = "curve_proxy", "US10Y + USDCNY parity" if a == "alt_us_bond" else "CN treasury 30Y yield"
            basis = "Fixed duration 7.5 carry+yield change, then FX parity return" if a == "alt_us_bond" else "Fixed duration 18.5 carry+yield change; lagged curve"
            action = "P1: certify approximation, availability and currency; not a tradable bond/futures return"
        else:
            source = source_ids.loc[a, "requested_code"]
            kind = "ETF_price" if source in {"511220.SH", "510300.SH", "159915.SZ", "159940.SZ", "510170.SH", "510880.SH", "518880.SH"} else "index_level"
            basis = "Observed price/level change; index total-return definition not automatically certified"
            action = "P0: certify distributions/splits" if kind == "ETF_price" else "P1: certify index name, level basis and history; no individual stock repair required"
            if a.startswith("credit_"):
                action += "; proxy does not establish literal issuer-sector/rating bucket"
            if a.startswith("alt_") and kind == "index_level":
                action += "; commodity index is not certified futures net PnL"
        row = dict(asset_code=a, asset_name=names[a], dependency_type=kind, source_identifier=source,
                   return_basis=basis, required_action=action, executable_nav_certified=False,
                   construction=contracts.loc[a, "construction"], **coverage(returns[a]))
        for j, h in enumerate((5, 20, 60)):
            row[f"valid_relative_labels_{h}d"] = int(label_masks[:, j, assets.index(a)].sum())
        inventory.append(row)
    inventory = pd.DataFrame(inventory)
    inventory.to_csv(OUT / "asset_sources_and_coverage.csv", index=False, encoding="utf-8-sig")
    inventory.loc[inventory.internal_missing_days.gt(0),
        ["asset_code", "asset_name", "internal_missing_days", "required_action"]].sort_values(
            "internal_missing_days", ascending=False).to_csv(
                OUT / "internal_gap_repair_priorities.csv", index=False, encoding="utf-8-sig")

    base = "data_processed/cb_evidence_v50/"
    balance = read(base + "last_known_balance_yuan.parquet")
    close = read(base + "quotes/close_individual.parquet").reindex_like(balance)
    size = read(base + "last_known_issue_size_yuan.parquet").reindex_like(close)
    observed = read(base + "individual_observed_price_returns.parquet").reindex_like(close)
    eligible = read("data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet").reindex_like(close)
    assert eligible.notna().all().all()
    volume = read(base + "quotes/volume_individual.parquet").reindex_like(close)
    excluded = read(base + "known_exit_exclusions.parquet").reindex_like(close)
    expected_eligible = (np.isfinite(close) & close.gt(0) & np.isfinite(volume) & volume.gt(0)
                         & np.isfinite(balance) & balance.gt(0) & np.isfinite(size) & size.gt(0)
                         & ~excluded.astype("boolean").fillna(True).astype(bool))
    assert np.array_equal(eligible, expected_eligible)
    bucket_returns = read("data_processed/cb_masked_research_v51/research_bucket_returns_not_executable_nav.parquet")
    assert returns.index.equals(close.index) and bucket_returns.index.equals(close.index)
    cb_names = [a for a in assets if a.startswith("cb_")]
    assert np.allclose(returns[cb_names], bucket_returns[cb_names], equal_nan=True)
    stats = {}
    missing = []
    fallback = []
    distinct = []
    replayed = 0
    for i, signal in enumerate(close.index[:-1]):
        day = close.index[i + 1]
        buckets = cb_bucket_weights(close.loc[signal].where(eligible.loc[signal]), balance.loc[signal], size.loc[signal])
        identities = []
        for name in cb_names:
            group = buckets.get(name)
            expected = bucket_returns.at[day, name]
            if group is None:
                assert pd.isna(expected)
                continue
            weights = group["weights"]
            got = realized_bucket_return(weights, observed.loc[day])
            assert (pd.isna(expected) and pd.isna(got)) or np.isclose(expected, got, atol=1e-12, rtol=1e-10)
            replayed += 1
            identities.append(tuple(zip(weights.index, weights.round(14))))
            fallback.append(dict(date=day, asset_code=name, fallback_level=group["fallback_level"], constituents=len(weights)))
            bad = ~np.isfinite(observed.loc[day, weights.index]) | observed.loc[day, weights.index].le(-1)
            for code, weight in weights.items():
                entry = stats.setdefault(code, dict(security=code, bucket_observation_count=0,
                    summed_bucket_weight=0., missing_bucket_observations=0, first_selected=signal, last_selected=signal,
                    buckets=set(), missing_dates=set()))
                entry["bucket_observation_count"] += 1
                entry["summed_bucket_weight"] += float(weight)
                entry["last_selected"] = signal
                entry["buckets"].add(name)
                if bad[code]:
                    entry["missing_bucket_observations"] += 1
                    entry["missing_dates"].add(day)
                    missing.append(dict(signal_date=signal, return_date=day, asset_code=name, security=code,
                                        precommitted_weight=float(weight), observed_return=observed.at[day, code]))
        distinct.append(dict(signal_date=signal, generated_buckets=len(identities), distinct_weight_vectors=len(set(identities))))
        if i % 400 == 0:
            print(f"V71 dependency replay {i}/{len(close)-1} signal={signal.date()}", flush=True)
    dep = []
    for entry in stats.values():
        entry = entry.copy()
        entry["distinct_bucket_count"] = len(entry.pop("buckets"))
        entry["missing_security_days"] = len(entry.pop("missing_dates"))
        dep.append(entry)
    dep = pd.DataFrame(dep).sort_values(["missing_security_days", "summed_bucket_weight"], ascending=False)
    dep.to_csv(OUT / "cb_actual_selected_security_dependencies.csv", index=False, encoding="utf-8-sig")
    missing = pd.DataFrame(missing, columns=["signal_date", "return_date", "asset_code", "security", "precommitted_weight", "observed_return"])
    missing.to_csv(OUT / "cb_selected_missing_observations.csv", index=False, encoding="utf-8-sig")
    fb = pd.DataFrame(fallback)
    fb.groupby(["asset_code", "fallback_level"]).size().rename("days").reset_index().to_csv(OUT / "cb_fallback_frequency.csv", index=False)
    pd.DataFrame(distinct).to_csv(OUT / "cb_daily_distinct_baskets.csv", index=False)
    pairs = []
    for i, a in enumerate(cb_names):
        for b in cb_names[i + 1:]:
            common = returns[[a, b]].dropna()
            pairs.append(dict(asset_a=a, asset_b=b, common_days=len(common),
                equal_return_days=int(np.isclose(common[a], common[b], atol=1e-12, rtol=0).sum()),
                correlation=float(common[a].corr(common[b])) if len(common) > 1 else None))
    pd.DataFrame(pairs).to_csv(OUT / "cb_pairwise_common_sample.csv", index=False)
    # Scope comparison only: the external strategy's ledger is not this project's holdings.
    queue_path = ROOT / "output/reports/cashflow_repair_v68/security_month_missing_ledger.csv"
    external_overlap = None
    if queue_path.exists():
        queue = read(str(queue_path.relative_to(ROOT)))
        code_column = next((c for c in ("ts_code", "security", "code") if c in queue), None)
        if code_column:
            external_overlap = len(set(queue[code_column]) & set(dep.security))

    kinds = inventory.dependency_type.value_counts().to_dict()
    summary = dict(dataset_sha256=digest(npz_path), start=str(returns.index.min().date()),
        end=str(returns.index.max().date()), rows=len(returns), assets=len(assets), dependency_types=kinds,
        return_mask_and_values_match=True, masked_zeros_are_not_observed_returns=True,
        v51_source_hashes_match=True, stored_signal_eligibility_matches_rule=True,
        selected_cb_securities=len(dep), input_cb_quote_columns=len(close.columns),
        cb_replayed_bucket_days=replayed, cb_replay_matches=True,
        cb_missing_selected_security_days=len(missing[["security", "return_date"]].drop_duplicates()),
        cb_missing_selected_unique_securities=int(missing.security.nunique()),
        cb_invalidated_bucket_days=len(missing[["asset_code", "return_date"]].drop_duplicates()),
        cb_fallback_fraction=float(fb.fallback_level.gt(0).mean()),
        cb_mean_distinct_baskets=float(pd.DataFrame(distinct).distinct_weight_vectors.mean()),
        external_queue_code_overlap_only=external_overlap,
        executable_nav_certified=False, promotion_allowed=False,
        synthetic=False, source_files_modified=False, market_data_downloaded=False,
        scope="Source and selected-membership audit, not a complete PIT/raw-vintage or execution certification")
    (OUT / "status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(inputs)}
    hashes[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    (OUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    short = inventory[["asset_code", "source_identifier", "first_valid", "last_valid", "internal_missing_days"]].to_markdown(index=False)
    text = f"""# 38资产数据依赖审计与修复范围（V71）

## 结论

不需要为了本项目，先修完另一套全市场动量策略的全部个券。也不能声称仅检查38列就足够：其中12列是自建转债篮子，必须追溯实际成分与现金流。

本次只读核查 V58 数据：{summary['start']} 至 {summary['end']}，{len(returns)} 个交易日、38个资产。来源分类为 {kinds}。没有下载行情、覆盖数据、重训或运行正式回测。

现存来源哈希与V51构建记录一致；全部日期的信号可买掩码与原规则重算一致。该检查验证实现一致性，不证明供应商历史可见时间已经认证。

## 必须修什么

1. **7个ETF**：逐一认证分红、拆分和价格口径；不需要重建ETF所持全部股票。511220拆分已在V58修正，但不等于全历史现金流已认证。
2. **16个指数序列**：核对具体指数身份、价格/财富口径和历史一致性；若只是研究指数本身，不逐只修复其全部成分。信用代理不能直接宣称精细的产业/城投与评级暴露；商品指数不等于可执行期货净收益。
3. **12个转债篮子**：按实际历史成员分批修复，不按当前名单、不按后来的收益筛选。此次重建实际选中过 {len(dep)} 只证券（输入报价表 {len(close.columns)} 列），验证 {replayed} 个篮子日收益与现有V51逐一一致。
4. **9只REITs组成的1列**：检查9个成分的分派与到账时间。现有除息日再投资研究序列不是支付日现金账，上市前不填历史。
5. **2个曲线代理**：30Y为久期18.5的一阶持有收益近似；美债为久期7.5的10Y曲线近似叠加人民币中间价变化。不是30Y ETF/TL或美债ETF实盘价格，应披露凸性、滚降、外汇可执行性与发布时滞限制。

## 其他实际缺口

城投AA代理 credit_lgfv_aa 有98个内部缺失日，不止V58主动屏蔽的一天；周期ETF equity_cyclical 有48日，美债曲线/汇率组合有24日，REITs有11日，创业板ETF有2日。缺失可能来自原始缺报价、合并/对齐或策略掩码，不能仅凭本表认定供应商没有数据，也不能统一前向填充成零收益。先沿对应来源定位，再决定补取或继续屏蔽。完整优先表见 internal_gap_repair_priorities.csv。

有色来源当前首个有效收益为2018-07-02，科创50来源为2020-07-24，REITs为2021-06-22。这些是当前数据首日，不自动等于产品上市日或指数发布日期；不得声称38资产全区间无缺。

## 转债的实际问题

- 实际选中成员出现 {summary['cb_missing_selected_security_days']} 个证券日缺口，涉及 {summary['cb_missing_selected_unique_securities']} 只证券，影响 {summary['cb_invalidated_bucket_days']} 个篮子日。这是优先查报价/停牌/退出证据的清单，**不是已经证明缺失的赎回事件数**。
- 当前实现遇到缺失成员会把整个篮子当日设为缺失，不把其收益当0、不用事后存活券重新归一。NPZ中的0只在掩码为False处作为存储占位，本次核对通过。
- 无缺报价的成员仍可能缺付息/赎回/转换现金流，不能仅修上面的缺口就认证总收益。正式组合还必须保留无法卖出库存，核验下一日成交与到账。
- “债性/平衡/股性”目前按价格 <=110、(110,130]、>130 划分，不是Delta或债底；“大/小”按可见发行规模的当日中位数，不是剩余市值。“高/低”按风格及规模组内价格中位数。
- 每组至少10只、剩余规模加权且单券上限10%。细组不足时按“细组→风格规模组→风格组→全体可用券”退回。退回占已生成篮子日 {summary['cb_fallback_fraction']:.1%}，每天平均只有 {summary['cb_mean_distinct_baskets']:.2f} 套不同的权重向量，12个名称不等于始终12种独立暴露。详细频率和共同样本相关性见CSV。
- 000832.CSI是独立转债参考指数，供相对标签与袖套基准使用，不是第39个资产，也没有替代12个篮子。

## 修复顺序与边界

先检查7个ETF现金流、信用两处已知异常的修复/掩码证据、9个REIT成分；并行按本报告实际转债依赖表核查成员缺口与公司行为。按历史选择权重和问题频率排序只是修复优先级，不据未来收益删券。

再完成自建转债篮子的付息、退出、库存、成交和成本账；随后才认证PPO奖励与组合净值。此次不放松门槛、不启动正式PPO、不发布新夏普。38资产与九宫格保留。

该报告不认证宏观发布时间、历史修订值、评级历史或完整原始数据时点性；需另行逐字段审查。文件哈希确保可复核输入，不证明经济口径正确。

## 数据覆盖

统一日历不等于所有资产都有2018年以来的真实数据。首个有效日、内部缺口及5/20/60日相对标签可用数详见 asset_sources_and_coverage.csv。不得倒填上市前价格或把研究代理称实物产品。

{short}

## 复核方式

运行 `python -X utf8 scripts/audit_v71_asset_dependencies.py`。输入不写回；输出仅在 output/reports/asset_dependency_v71。source_hashes.json 记录输入与脚本哈希，status.json 记录检查结果。缺口明细中的权重是各篮子的权重，不能跨篮子相加后冒充组合损失或NAV覆盖率。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
