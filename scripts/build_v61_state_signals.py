"""Read-only V58 inputs -> named research baselines; never fabricate Boundary."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v61_state_momentum import price_rule_signals


def main():
    out = ROOT / "output/reports/state_momentum_v61"
    data = ROOT / "data_processed/state_momentum_v61"
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data_processed/corporate_action_v58/dataset.npz"
    with np.load(source) as z:
        a = {k:z[k] for k in z.files}
    meta = json.loads((ROOT / "output/reports/corporate_action_v58/metadata.json").read_text())
    dates = pd.to_datetime(a["dates"])
    returns = pd.DataFrame(np.where(a["return_masks"], a["returns"], np.nan), index=dates, columns=meta["assets"])
    signals = price_rule_signals(returns, a["module_index"])
    arrays = {k:f.to_numpy(np.float32) for k,f in signals.items()}
    arrays["dates"] = a["dates"]
    np.savez_compressed(data / "signals.npz", **arrays)
    records = []
    for name,f in signals.items():
        for code in f:
            records.append(dict(signal=name, asset=code, known_days=int(f[code].notna().sum()),
                first_known=str(f[code].first_valid_index()) if f[code].notna().any() else None,
                last_known=str(f[code].last_valid_index()) if f[code].notna().any() else None,
                original_asset_retained=True, formal_execution=False))
    pd.DataFrame(records).to_csv(out / "signal_coverage.csv", index=False)
    report = dict(assets=38, source_dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        signals_sha256=hashlib.sha256((data / "signals.npz").read_bytes()).hexdigest(),
        value_boundary_enabled=False, value_boundary_reason="CSI300 historical dividend yield definition and PIT availability not verified",
        ifind_probe="HTTP200 but requested12 rows vs returned335; historical field metadata and calendar-filled dates unresolved; not ingested",
        price_extension_rule_is_not_value_boundary=True, signal_is_not_cash_weight=True,
        original_data_modified=False, formal_promotion=False, ppo_training_allowed=False)
    (out / "signal_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
