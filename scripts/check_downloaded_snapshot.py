"""Read-only snapshot readiness check; never certifies financial performance."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = []
    for name in ['asset_returns_daily.parquet', 'asset_returns_monthly.parquet',
                 'feature_panel_daily_v43.parquet', 'feature_panel_monthly_v43.parquet']:
        path = ROOT / 'data_processed' / name
        if not path.is_file():
            raise FileNotFoundError('Restore Release assets first: ' + str(path))
        table = pd.read_parquet(path)
        reports.append(dict(file=name, rows=len(table), columns=len(table.columns)))
    status_path = ROOT / 'output/reports/cb_cashflows_v95/status.json'
    state = json.loads(status_path.read_text(encoding='utf-8'))
    print(json.dumps(dict(data_files_readable=reports, accounting_replay_days=state['completed_days'],
                          formal_training_allowed=state['formal_training_allowed'],
                          certified_total_return=state['certified_total_return'],
                          note='Readiness is not a repaired strategy-performance certificate.'), indent=2))


if __name__ == '__main__':
    main()
