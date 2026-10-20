"""Broaden search terms only; exact bond identity still checked inside each PDF."""
from pathlib import Path
import hashlib
import json
import re
import sys
import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [("127016.SZ", "鲁泰", "2026-03-25~2026-04-16", ["鲁泰转债"]),
         ("110070.SH", "凌钢", "2026-04-01~2026-04-20", ["凌钢转债"]),
         ("123049.SZ", "维尔利", "2026-04-01~2026-04-20", ["维尔转债", "维尔利转债"])]


def main():
    raw = ROOT / "data_raw/inventory_terminal_v87/issuer_fallback"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
    manifest = []
    for code, keyword, period, aliases in CASES:
        folder = raw / code
        folder.mkdir(exist_ok=True)
        announcements = []
        for page in range(1, 6):
            path = folder / ("index.json" if page == 1 else f"index_page_{page}.json")
            if not path.exists():
                response = session.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=dict(
                    pageNum=page, pageSize=30, column="sse" if code.endswith(".SH") else "szse",
                    tabName="fulltext", stock="", searchkey=keyword, secid="", plate="", category="",
                    trade="", seDate=period, sortName="time", sortType="asc", isHLtitle="false"), timeout=(10, 35))
                response.raise_for_status()
                payload = response.json()
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
            announcements.extend(payload.get("announcements") or [])
            if not payload.get("hasMore"):
                break
        else:
            raise ValueError("Issuer query exceeds five-page evidence budget")
        if len(announcements) != len({r["announcementId"] for r in announcements}):
            raise ValueError("Duplicate announcement across archive pages")
        for row in announcements:
            title = re.sub("<[^>]+>", "", row.get("announcementTitle", ""))
            compact_title = "".join(title.split())
            if not any(alias in compact_title for alias in aliases) or not any(t in title for t in ["兑付", "赎回"]):
                continue
            if any(t in title for t in ["不提前", "核查", "提示"]):
                continue
            kind = "result" if "结果" in title or "股份变动" in title or "股本变动" in title else "schedule"
            relative = row["adjunctUrl"]
            if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/\d+\.[Pp][Dd][Ff]", relative):
                raise ValueError("Unexpected archive path")
            pdf = folder / Path(relative).name
            url = "https://static.cninfo.com.cn/" + relative
            if not pdf.exists():
                response = session.get(url, timeout=(10, 40))
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("Not a PDF")
                pdf.write_bytes(response.content)
            text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
            if code[:6] not in "".join(text.split()):
                raise ValueError("PDF bond identity mismatch")
            pdf.with_suffix(".txt").write_text(text, encoding="utf-8")
            item = dict(ts_code=code, name=keyword, kind=kind, title=title, url=url,
                        publication_date=relative.split("/")[1], pdf=str(pdf.relative_to(ROOT)),
                        sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), approved_for_ledger=False)
            manifest.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    (raw / "candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
