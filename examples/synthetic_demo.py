"""Offline synthetic data smoke test, not a strategy reproduction."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def main():
    output = Path(__file__).resolve().parent / 'generated'
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261020)
    dates = pd.bdate_range('2020-01-01', periods=300)
    columns = [f'synthetic_asset_{i:02d}' for i in range(1, 39)]
    common = rng.normal(0, 0.001, (len(dates), 1))
    returns = pd.DataFrame(common + rng.normal(0, 0.003, (len(dates), 38)), index=dates, columns=columns)
    returns.index.name = 'date'
    prices = 100 * (1 + returns).cumprod()
    weights = np.repeat(1 / 38, 38)
    portfolio = returns @ weights
    assert returns.shape == (300, 38)
    assert np.isfinite(returns.to_numpy()).all()
    assert np.isclose(weights.sum(), 1)
    returns.to_csv(output / 'SYNTHETIC_returns_daily.csv')
    prices.to_csv(output / 'SYNTHETIC_prices_daily.csv')
    summary = dict(synthetic=True, real_market_data=False, strategy_backtest=False,
                   observations=len(dates), series=38,
                   equal_weight_demo_final_nav=float((1 + portfolio).prod()))
    (output / 'SYNTHETIC_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
