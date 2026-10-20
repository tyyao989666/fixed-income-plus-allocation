"""A dated issuer delisting-decision notice, separate from later recovery plans."""
from pathlib import Path
import fetch_v89_distressed_evidence as fetch


if __name__ == "__main__":
    fetch.RAW = Path(__file__).resolve().parents[1] / "data_raw/delisting_trigger_v94"
    fetch.CASES = [("123099.SZ", "2025-04-18",
        "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10908627&stockid=300630",
        ["123099", "终止上市的决定", "2025年4月18日", "2025年4月28日"])]
    fetch.main()
