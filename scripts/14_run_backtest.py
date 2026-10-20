from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import load_config, run_backtest


if __name__ == "__main__":
    cfg = load_config()
    nav, metrics = run_backtest(cfg)
    total_return = metrics.loc[metrics["metric"].eq("total_return"), "value"].iloc[0]
    max_drawdown = metrics.loc[metrics["metric"].eq("max_drawdown"), "value"].iloc[0]
    print(
        "v1 monthly backtest generated: "
        f"months={len(nav)}, total_return={total_return:.4f}, max_drawdown={max_drawdown:.4f}"
    )
