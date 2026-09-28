"""Fetch the source PDFs from nvlpubs.nist.gov (public, no login). Idempotent."""

from __future__ import annotations

import urllib.request

from . import config

UA = "Mozilla/5.0 (portfolio research project; +https://github.com/Waleed-Ilyas)"


def fetch_all(force: bool = False) -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    for doc in config.DOCS:
        dest = config.RAW / doc.filename
        if dest.exists() and dest.stat().st_size > 50_000 and not force:
            continue
        print(f"downloading {doc.url}")
        req = urllib.request.Request(doc.url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        with open(dest, "rb") as f:
            if f.read(5) != b"%PDF-":
                dest.unlink()
                raise RuntimeError(f"{doc.url} did not return a PDF")
