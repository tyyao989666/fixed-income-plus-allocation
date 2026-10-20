"""Bounded primary-announcement acquisition for actually frozen CB inventory."""
from pathlib import Path
import hashlib
import json
import re
import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw/inventory_terminal_v84"
CASES = [("113507.SH", "天马转债", "2019-10-25~2019-11-25"),
         ("110030.SH", "格力转债", "2019-12-01~2019-12-31")]


def main(cases=CASES, raw=RAW, selected_ids=None):
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
    manifest = []
    for code, name, period in cases:
        folder = raw / code
        folder.mkdir(exist_ok=True)
        query = folder / "index.json"
        if not query.exists():
            response = session.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=dict(
                pageNum=1, pageSize=30, column="sse" if code.endswith(".SH") else "szse", tabName="fulltext", stock="", searchkey=name,
                secid="", plate="", category="", trade="", seDate=period, sortName="time",
                sortType="asc", isHLtitle="false"), timeout=(10, 35))
            response.raise_for_status()
            payload = response.json()
            query.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            payload = json.loads(query.read_text(encoding="utf-8"))
        if payload.get("hasMore"):
            raise ValueError("Bounded announcement query requires pagination; refuse silent truncation")
        candidates = []
        for row in payload.get("announcements") or []:
            title = re.sub("<[^>]+>", "", row.get("announcementTitle", ""))
            forced = (selected_ids or {}).get((code, row.get("announcementId")))
            if forced:
                candidates.append((forced, row, title))
                continue
            if name not in title or not any(t in title for t in ["赎回", "兑付"]):
                continue
            if "结果" in title:
                candidates.append(("result", row, title))
            elif "摘牌" in title or not any(x in title for x in ["提示", "停止", "再次", "第三"]):
                candidates.append(("schedule", row, title))
        if len(candidates) > 6:
            raise ValueError("Ambiguous candidates require review")
        for kind, row, title in candidates:
            relative = row["adjunctUrl"]
            if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/\d+\.[Pp][Dd][Ff]", relative):
                raise ValueError("Unexpected announcement path")
            url = "https://static.cninfo.com.cn/" + relative
            path = folder / Path(relative).name
            if not path.exists():
                response = session.get(url, timeout=(10, 40))
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("Not a PDF")
                path.write_bytes(response.content)
            pages = [p.extract_text() or "" for p in PdfReader(path).pages]
            path.with_suffix(".txt").write_text("\n".join(pages), encoding="utf-8")
            compact = "".join("".join(pages).split())
            if code[:6] not in compact:
                raise ValueError("Security identifier absent from candidate")
            item = dict(ts_code=code, name=name, kind=kind, title=title, url=url,
                publication_date=relative.split("/")[1], pdf=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), approved_for_ledger=False)
            manifest.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    (raw / "candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
