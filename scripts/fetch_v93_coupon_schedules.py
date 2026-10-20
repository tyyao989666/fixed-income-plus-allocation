"""Two high-exposure ordinary coupon schedules; no inferred cash receipts."""
from pathlib import Path
import fetch_v89_distressed_evidence as fetch


if __name__ == "__main__":
    fetch.RAW = Path(__file__).resolve().parents[1] / "data_raw/cb_coupons_v93"
    fetch.CASES = [
        ("electric_2019", "2019-01-28", "https://www.hkexnews.hk/listedco/listconews/sehk/2019/0128/LTN20190128419_C.pdf", ["113008", "2019年2月1日", "2019年2月11日", "2019年2月15日"]),
        ("spdb_2024", "2024-10-21", "https://static.cninfo.com.cn/finalpage/2024-10-21/1221442730.PDF", ["110059", "2024年10月25日", "2024年10月28日", "3.2"]),
    ]
    fetch.main()
