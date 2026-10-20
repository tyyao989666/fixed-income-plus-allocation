from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.v50_cb_evidence import clean_number,quote_table,issuance_registry,fill_preconversion_known_balance,requested_codes


class SourceEvidence(unittest.TestCase):
    def test_codes_adjacent_to_chinese_names(self):
        self.assertEqual(requested_codes("113050.SH南银转债、113065.SH齐鲁转债"),["113050.SH","113065.SH"])
    def test_only_table_numbers(self):
        self.assertEqual(clean_number("12.3万"),123000)
        self.assertTrue(np.isnan(clean_number("大约12.3万")))

    def sample(self):
        return pd.DataFrame({"证券代码":["A","A"],"日期":[20260102,20260105],
            "收盘价（单位：元）":[101,102],"前收盘价（单位：元）":[100,101],"成交量":[0,"12万"]})

    def test_date_code_and_zero_volume(self):
        f=quote_table(self.sample(),["A"],"2026-01-01","2026-01-04")
        self.assertEqual(len(f),1);self.assertFalse(f.positive_volume.iloc[0])
        self.assertTrue(quote_table(self.sample(),["B"],"2026-01-01","2026-01-06").empty)

    def test_current_status_without_date_is_not_a_quote(self):
        self.assertTrue(quote_table(pd.DataFrame({"证券代码":["A"],"交易状态":["终止上市"]}),["A"],"2020-01-01","2020-01-10").empty)

    def test_valuation_not_accepted_as_transaction_close(self):
        f=self.sample().rename(columns={"收盘价（单位：元）":"估值净价（单位：元）"})
        self.assertTrue(quote_table(f,["A"],"2026-01-01","2026-01-06").empty)

    def registry(self,result="20260102"):
        dates=pd.bdate_range("2026-01-01",periods=10)
        m=pd.DataFrame(dict(issue_size=[1e9],list_date_parsed=[pd.Timestamp("2026-01-06")],conv_start_date=["20260109"]),index=["A"])
        issues=pd.DataFrame(dict(ts_code=["A"],ann_date=["20260101"],res_ann_date=[result],issue_size=[10.],issue_type=["优先配售和网上定价"]))
        return dates,issuance_registry(issues,m,dates)

    def test_mixed_vendor_units_verified_against_issue_master(self):
        _,r=self.registry();self.assertEqual(r.loc["A","unit_factor"],1e8)
        self.assertEqual(r.loc["A","issue_size_yuan"],1e9)

    def test_baseline_only_after_result_and_before_conversion(self):
        d,r=self.registry()
        k={name:pd.DataFrame(np.nan,index=d,columns=["A"]) for name in ["balance","issue_size","age_days"]}
        out,src=fill_preconversion_known_balance(k,r,d)
        self.assertTrue(out["balance"].loc[:"2026-01-02"].isna().all().all())
        self.assertEqual(out["balance"].loc["2026-01-05","A"],1e9)
        self.assertTrue(out["balance"].loc["2026-01-09":].isna().all().all())

    def test_future_issue_result_never_fills_past(self):
        d,r=self.registry("20260112")
        k={name:pd.DataFrame(np.nan,index=d,columns=["A"]) for name in ["balance","issue_size","age_days"]}
        out,_=fill_preconversion_known_balance(k,r,d)
        self.assertTrue(out["balance"].isna().all().all())

    def test_real_disclosure_not_overwritten(self):
        d,r=self.registry()
        k={name:pd.DataFrame(np.nan,index=d,columns=["A"]) for name in ["balance","issue_size","age_days"]}
        k["balance"].loc["2026-01-06","A"]=9e8
        out,_=fill_preconversion_known_balance(k,r,d)
        self.assertEqual(out["balance"].loc["2026-01-06","A"],9e8)

    def test_old_issue_100m_rounding_is_not_a_unit_failure(self):
        dates=pd.bdate_range("2018-01-01",periods=10)
        master=pd.DataFrame(dict(issue_size=[874723000.],list_date_parsed=[dates[5]],conv_start_date=["20180701"]),index=["A"])
        issues=pd.DataFrame(dict(ts_code=["A"],ann_date=["20180101"],res_ann_date=["20180102"],issue_size=[8.7472],issue_type=["网上定价"]))
        r=issuance_registry(issues,master,dates)
        self.assertEqual(r.loc["A","unit_factor"],1e8)
        self.assertEqual(r.loc["A","size_rounding_error_yuan"],-3000.)


if __name__=="__main__":unittest.main()
