"""Acquire bounded issuer/trustee documents, without approving recoveries."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw/distressed_claims_v89"
CASES = [
    ("128114.SZ", "2024-01-31", "https://static.cninfo.com.cn/finalpage/2024-01-31/1219044123.PDF", ["128114", "100.14"]),
    ("127033.SZ", "2026-02-11", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=11961378&stockid=002822", ["127033", "50,315"]),
    ("110072.SH", "2026-07-29", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12468315&stockid=600297", ["404004", "重组"]),
    ("123015.SZ", "2026-01-07", "https://quotes.sina.com.cn/bond/view/announcement_show.php?id=23941479", ["404001", "违约"]),
    ("128100.SZ", "2025-03-11", "https://static.cninfo.com.cn/finalpage/2025-03-11/1222768991.PDF", ["404002", "破产"]),
]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    manifest = []
    for code, published, url, tokens in CASES:
        folder = RAW / code
        folder.mkdir(exist_ok=True)
        row = dict(ts_code=code, publication_date=published, url=url,
                   approved_for_cash=False, publication_date_verified=False)
        try:
            pdf_url = url
            if not url.lower().endswith(".pdf"):
                page = folder / "source.html"
                if not page.exists():
                    response = session.get(url, timeout=(10, 35))
                    response.raise_for_status()
                    page.write_bytes(response.content)
                soup = BeautifulSoup(page.read_bytes(), "html.parser")
                links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)
                         if re.search(r"\.pdf(?:$|\?)", a["href"], re.I)]
                if len(set(links)) != 1:
                    raise ValueError(f"Expected one original PDF, got {len(set(links))}")
                pdf_url = links[0]
                row["html_sha256"] = hashlib.sha256(page.read_bytes()).hexdigest()
            pdf = folder / "source.pdf"
            if not pdf.exists():
                response = session.get(pdf_url, timeout=(10, 40))
                response.raise_for_status()
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("Not a PDF")
                pdf.write_bytes(response.content)
            pages = [p.extract_text() or "" for p in PdfReader(pdf).pages]
            content = "\n".join(pages)
            compact = "".join(content.split()).replace(",", "").replace("，", "")
            pdf.with_suffix(".txt").write_text(content, encoding="utf-8")
            if not all(t.replace(",", "") in compact for t in tokens):
                raise ValueError("Required security/contract tokens absent")
            row.update(status="original_pdf_acquired_identity_checked", pdf_url=pdf_url,
                       pdf=str(pdf.relative_to(ROOT)), pages=len(pages),
                       sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
        except (requests.RequestException, ValueError) as exc:
            row.update(status="acquisition_or_identity_blocked", error=str(exc))
        manifest.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (RAW / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
