"""Audit the authorized six-name screen; no performance certification."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_v92_excluded_cb_universe as auditor

OUT = ROOT / 'output/reports/cb_cashflows_v95'
DATA = ROOT / 'data_processed/cb_cashflows_v95'


def main():
    auditor.OUT, auditor.DATA = OUT, DATA
    auditor.main()
    state = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
    policy = json.loads((ROOT / 'config/cb_research_exclusions_v95.json').read_text(encoding='utf-8'))
    assert state['excluded_securities'] == policy['excluded_securities']
    assert len(set(state['excluded_securities'])) == 6
    assert state['asset_structure_count'] == 38 and state['basket_count'] == 12
    assert state['universe_after'] == state['universe_before'] - 6
    daily = pd.read_parquet(DATA / 'daily_accounting_diagnostics.parquet')
    final = pd.read_parquet(DATA / 'final_quantities.parquet')
    assert daily.asset_code.nunique() == 12
    assert daily.date.nunique() == state['total_days'] == state['completed_days']
    assert not final[policy['excluded_securities']].to_numpy().any()
    assert np.isfinite(daily[['settlement_cash_yuan', 'receivable_yuan', 'cost_yuan']].to_numpy()).all()
    before = json.loads((ROOT / 'output/reports/cb_cashflows_v94/status.json').read_text(encoding='utf-8'))
    result = dict(passed=True, excluded_securities=policy['excluded_securities'],
                  all_six_excluded_before_basket_construction=True,
                  asset_structure_count=38, basket_count=12,
                  completed_days=state['completed_days'],
                  previous_missing_position_days=before['unresolved_position_days'],
                  missing_position_days=state['unresolved_position_days'],
                  ending_unresolved_securities=state['ending_unresolved_securities'],
                  ending_receivable_yuan=state['ending_receivable_yuan'],
                  raw_data_deleted=False, formal_promotion=False,
                  full_three_product_backtest_completed=False, training_started=False,
                  retrospective_universe_selection=True,
                  source_sha256={str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT / 'six_name_exclusion_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
