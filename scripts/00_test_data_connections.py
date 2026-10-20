from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare as ts

token = ts.get_token()
print({"tushare_token_found": bool(token), "wind_status": "not_called_in_data_prep_stage"})
