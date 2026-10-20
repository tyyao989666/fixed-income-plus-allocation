from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep_pipeline import run_all

run_all(ROOT)
print("asset_universe.parquet rebuilt")
