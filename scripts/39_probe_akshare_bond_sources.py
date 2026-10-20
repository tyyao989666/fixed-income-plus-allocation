from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "reports"
RAW = ROOT / "data_raw" / "akshare"


def main() -> None:
    funcs = [
        "bond_corporate_issue_cninfo",
        "bond_debt_nafmii",
        "macro_china_bond_public",
        "macro_china_new_financial_credit",
        "bond_cash_summary_sse",
        "bond_deal_summary_sse",
        "bond_new_composite_index_cbond",
    ]
    rows = []
    RAW.mkdir(parents=True, exist_ok=True)
    for f in funcs:
        try:
            obj = getattr(ak, f)
            df = obj()
            path = RAW / f"{f}.parquet"
            df.to_parquet(path, index=False)
            df.head(1000).to_csv(RAW / f"{f}_head.csv", index=False, encoding="utf-8-sig")
            rows.append(
                {
                    "function": f,
                    "status": "ok",
                    "rows": len(df),
                    "columns": ",".join(map(str, df.columns)),
                    "file": str(path.relative_to(ROOT)),
                }
            )
        except Exception as exc:
            rows.append({"function": f, "status": "failed", "rows": 0, "columns": "", "file": "", "error": str(exc)[:500]})
    report = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    report.to_excel(REPORTS / "akshare_bond_source_probe_report.xlsx", index=False)
    report.to_csv(REPORTS / "akshare_bond_source_probe_report.csv", index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
