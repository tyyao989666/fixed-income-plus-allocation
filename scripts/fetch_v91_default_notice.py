"""Archive a subsequent issuer default notice; no par repayment event."""
from pathlib import Path
import fetch_v89_distressed_evidence as fetch


if __name__ == "__main__":
    fetch.RAW = Path(__file__).resolve().parents[1] / "data_raw/distressed_default_v91"
    fetch.CASES = [
        ("guanghui_default", "2026-08-17", "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12499999&stockid=600297",
         ["404004", "本息无法按期兑付", "2026年8月17日"]),
    ]
    fetch.main()
