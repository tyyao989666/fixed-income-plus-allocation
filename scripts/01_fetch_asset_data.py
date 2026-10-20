from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep_pipeline import ensure_dirs, fetch_configured_assets, fetch_extra_raw_data, load_yaml

paths = ensure_dirs(ROOT)
config = load_yaml(paths.config / "asset_universe.yaml")
extra_status = fetch_extra_raw_data(paths, config)
universe, daily = fetch_configured_assets(paths, config)
universe.to_parquet(paths.processed / "_asset_universe_config_fetch.parquet", index=False)
daily.to_parquet(paths.processed / "_asset_returns_config_fetch.parquet", index=False)
print({"asset_rows": len(universe), "daily_rows": len(daily), "extra": extra_status})
