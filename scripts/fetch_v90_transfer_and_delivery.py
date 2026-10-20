"""Acquire new primary evidence without changing the bound V89 fetcher."""
from pathlib import Path
import fetch_v89_distressed_evidence as fetch


if __name__ == "__main__":
    fetch.RAW = Path(__file__).resolve().parents[1] / "data_raw/distressed_claims_v90"
    fetch.CASES = [
        ("blue_transfer", "2023-09-14", "https://static.cninfo.com.cn/finalpage/2023-09-14/1217859882.PDF", ["123015", "404001", "2023年9月18日"]),
        ("soute_transfer", "2023-10-18", "https://static.cninfo.com.cn/finalpage/2023-10-18/1218075143.PDF", ["128100", "404002", "2023年10月20日"]),
        ("guanghui_bridge", "2024-10-30", "https://static.cninfo.com.cn/finalpage/2024-10-30/1221570114.PDF", ["110072", "404004", "2024年11月1日"]),
        ("zhongzhuang_delivery", "2026-08-28", "https://disc.static.szse.cn/disc/disk03/finalpage/2026-08-28/5f15fa4a-0016-44bf-8732-18b950353a7a.PDF", ["127033", "50,315", "2026年2月12日"]),
        ("zhengbang_delivery", "2025-04-25", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10969454&stockid=002157", ["128114", "全额清偿本金及利息并摘牌", "2024年1月31日"]),
    ]
    fetch.main()
