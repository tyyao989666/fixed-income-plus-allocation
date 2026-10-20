"""Build isolated masked CB research labels. Never replaces formal model data."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v49_point_in_time import cb_bucket_weights, realized_bucket_return
from src.v51_cb_policy import buy_eligibility, masked_forward_labels


def main():
    source = ROOT / "data_processed/cb_evidence_v50"
    out = ROOT / "data_processed/cb_masked_research_v51"
    report = ROOT / "output/reports/cb_masked_research_v51"
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    balance = pd.read_parquet(source / "last_known_balance_yuan.parquet")
    close = pd.read_parquet(source / "quotes/close_individual.parquet").reindex_like(balance)
    volume = pd.read_parquet(source / "quotes/volume_individual.parquet").reindex_like(close)
    size = pd.read_parquet(source / "last_known_issue_size_yuan.parquet").reindex_like(close)
    excluded = pd.read_parquet(source / "known_exit_exclusions.parquet").reindex_like(close)
    observed = pd.read_parquet(source / "individual_observed_price_returns.parquet").reindex_like(close)
    with (ROOT / "output/reports/v45_metadata.json").open(encoding="utf-8") as stream:
        context = json.load(stream)["assets"]
    names = [x for x in context if x.startswith("cb_")]
    assert len(context) == 38 and len(names) == 12
    eligible = pd.DataFrame(False, index=close.index, columns=close.columns)
    reasons_count = []
    returns = pd.DataFrame(np.nan, index=close.index, columns=names)
    records = []
    for i, signal in enumerate(close.index):
        allowed, reasons = buy_eligibility(close.loc[signal], volume.loc[signal],
            balance.loc[signal], size.loc[signal], excluded.loc[signal])
        eligible.loc[signal] = allowed
        reasons_count.append(dict(signal_date=signal, eligible=int(allowed.sum()), **reasons.sum().astype(int).to_dict()))
        if i + 1 == len(close.index):
            continue
        day = close.index[i + 1]
        buckets = cb_bucket_weights(close.loc[signal].where(allowed), balance.loc[signal], size.loc[signal])
        for name in names:
            group = buckets.get(name)
            if group is None:
                continue
            w = group["weights"]
            returns.at[day, name] = realized_bucket_return(w, observed.loc[day])
            records.append(dict(date=day, signal_date=signal, asset_code=name,
                constituents=len(w), maximum_weight=float(w.max()), weight_sum=float(w.sum()),
                fallback_level=group["fallback_level"], missing_components=int(observed.loc[day, w.index].isna().sum())))
        if i % 400 == 0:
            print(f"V51 snapshot {i}/{len(close.index)}, {signal.date()}", flush=True)
    eligible.to_parquet(out / "signal_buy_eligibility.parquet")
    returns.to_parquet(out / "research_bucket_returns_not_executable_nav.parquet")
    values, masks = masked_forward_labels(returns)
    np.savez_compressed(out / "cb_forward_labels.npz", dates=close.index.asi8,
        assets=np.asarray(names), horizons=np.asarray([5, 20, 60]), targets=values, target_masks=masks)
    pd.DataFrame(records).to_csv(report / "prior_signal_bucket_audit.csv", index=False)
    pd.DataFrame(reasons_count).to_csv(report / "signal_eligibility_counts.csv", index=False)
    coverage = pd.DataFrame([dict(asset_code=name, horizon=h, valid_labels=int(masks[:, j, a].sum()),
        total_dates=len(close), missing_return_days=int(returns[name].isna().sum()))
        for a, name in enumerate(names) for j, h in enumerate([5, 20, 60])])
    coverage.to_csv(report / "label_coverage.csv", index=False)
    manifest = dict(status="isolated_research_label_pack_built", start=str(close.index[0].date()),
        end=str(close.index[-1].date()), calendar_rows=len(close), context_assets=context,
        cb_assets=names, missing_bucket_days=int(returns.isna().sum().sum()),
        formal_dataset_replaced=False, models_retrained=False, portfolio_promoted=False,
        executable_inventory_replay_completed=False,
        reference_policy="Absolute forward labels only; relative labels must retain fixed-at-signal reference dependencies. No future renormalization.",
        limitations=["Daily reconstituted research returns are not executable inventory NAV",
            "Coupon and terminal cashflows not yet certified", "Remaining 26 assets not refreshed by this runner",
            "Positive signal volume is historical evidence, not a guarantee of next-day execution"],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    inputs = [source / name for name in ["last_known_balance_yuan.parquet", "last_known_issue_size_yuan.parquet",
        "known_exit_exclusions.parquet", "individual_observed_price_returns.parquet",
        "quotes/close_individual.parquet", "quotes/volume_individual.parquet"]]
    inputs += [ROOT / "output/reports/v45_metadata.json", ROOT / "src/v51_cb_policy.py",
               ROOT / "src/v49_point_in_time.py"]
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in inputs}
    (report / "status.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("context_assets", "limitations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
