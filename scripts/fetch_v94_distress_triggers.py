"""Bounded exact-date primary announcement lookup, no stock/bond price fallback."""
from pathlib import Path
import hashlib
import json
import re

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw/distress_triggers_v94"
CASES = [("128114.SZ", "正邦", "2023-07-21"),
         ("113578.SH", "全筑", "2023-11-14"),
         ("127033.SZ", "中装", "2025-08-20")]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
    records = []
    for code, name, day in CASES:
        folder = RAW / code
        folder.mkdir(exist_ok=True)
        index = folder / "index.json"
        if not index.exists():
            response = session.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=dict(
                pageNum=1, pageSize=30, column="sse" if code.endswith(".SH") else "szse", tabName="fulltext",
                stock="", searchkey=name, secid="", plate="", category="", trade="", seDate=day+"~"+day,
                sortName="time", sortType="asc", isHLtitle="false"), timeout=(10, 35))
            response.raise_for_status()
            index.write_text(json.dumps(response.json(), ensure_ascii=False, indent=2), encoding="utf-8")
        payload = json.loads(index.read_text(encoding="utf-8"))
        if payload.get("hasMore"):
            raise ValueError("Unexpected pagination; refuse truncated search")
        candidates = []
        for row in payload.get("announcements") or []:
            title = re.sub("<[^>]+>", "", row["announcementTitle"])
            if "法院裁定受理" in title and "公司重整" in title and "子公司" not in title:
                candidates.append((row, title))
        if len(candidates) != 1:
            raise ValueError(f"Need one exact primary trigger for {code}, got {len(candidates)}")
        row, title = candidates[0]
        url = "https://static.cninfo.com.cn/" + row["adjunctUrl"]
        pdf = folder / "trigger.pdf"
        if not pdf.exists():
            response = session.get(url, timeout=(10, 40))
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("Not PDF")
            pdf.write_bytes(response.content)
        text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
        pdf.with_suffix(".txt").write_text(text, encoding="utf-8")
        compact = "".join(text.split())
        if code[:6] not in compact or "受理" not in compact or "重整" not in compact:
            raise ValueError("Bond identifier or trigger words absent")
        if row["adjunctUrl"].split("/")[1] != day:
            raise ValueError("Publication date conflicts with exact-day lookup")
        item = dict(ts_code=code, publication_date=day, trigger_type="court_accepts_issuer_reorganization",
                    title=title, url=url, pdf=str(pdf.relative_to(ROOT)), sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
        records.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (RAW / "trigger_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
