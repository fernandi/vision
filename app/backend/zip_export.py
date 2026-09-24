"""
Collection → ZIP export. Stdlib only, so dev_server.py can reuse it.

Files are named "07 - Author - Title (Museum).jpg" in collection order, and a
credits.csv lists every work with its museum page and licence.
"""
import csv
import io
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

MAX_ITEMS = 200
MUSEUMS = {
    "MET": "The Met",
    "CMA": "Cleveland Museum of Art",
    "artic": "Art Institute of Chicago",
    "Smithonian": "Smithsonian",
}
EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp",
              "image/gif": "gif", "image/tiff": "tif"}
UA = {"User-Agent": "Mozilla/5.0 (Glane collection export)"}


def fetch(url: str, timeout: int = 30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def _clean(text: str, limit: int) -> str:
    text = re.sub(r"\s*\([^)]*\)", "", text or "")          # "(American, 1824–1906)"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)       # forbidden in file names
    text = re.sub(r"\s+", " ", text).strip(" .")
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(" .,;") + "…"
    return text


def _extension(content_type: str, url: str) -> str:
    ext = EXTENSIONS.get(content_type.split(";")[0].strip().lower())
    if ext:
        return ext
    m = re.search(r"\.(jpe?g|png|webp|gif|tiff?)(?:$|\?)", url.lower())
    return (m.group(1).replace("jpeg", "jpg") if m else "jpg")


def safe_name(name: str) -> str:
    return _clean(name, 60) or "collection"


def file_name(index: int, width: int, item: dict, ext: str) -> str:
    author = item.get("Author") or ""
    parts = [str(index).zfill(width)]
    if author and author != "N/A":
        parts.append(_clean(author, 40))
    parts.append(_clean(item.get("Title") or "Untitled", 80))
    museum = MUSEUMS.get(item.get("source", ""), item.get("source", ""))
    return f"{' - '.join(p for p in parts if p)} ({museum}).{ext}"


def build_zip(items: list, fetcher=fetch):
    """items: metadata dicts with image_url (+ original_url fallback), Title, Author, source, URL.
    Returns (zip bytes, number of images that could not be fetched)."""
    items = items[:MAX_ITEMS]

    def grab(item):
        # Mirrored HD copy first, then the museum's own file.
        for url in dict.fromkeys(u for u in (item.get("image_url"), item.get("original_url")) if u):
            try:
                return url, *fetcher(url)
            except Exception:
                continue
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        downloads = list(pool.map(grab, items))

    width = max(2, len(str(len(items))))
    buf = io.BytesIO()
    credits = io.StringIO()
    writer = csv.writer(credits)
    writer.writerow(["file", "title", "author", "museum", "museum_page", "image_url", "licence"])
    missing = 0
    used = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:   # images are already compressed
        for i, (item, got) in enumerate(zip(items, downloads), start=1):
            museum = MUSEUMS.get(item.get("source", ""), item.get("source", ""))
            name = ""
            if got:
                url, data, ctype = got
                name = file_name(i, width, item, _extension(ctype, url))
                if name in used:
                    name = name.replace(" (", f" [{i}] (", 1)
                used.add(name)
                zf.writestr(name, data)
            else:
                missing += 1
            writer.writerow([name or "(unavailable)", item.get("Title", ""), item.get("Author", ""),
                             museum, item.get("URL", ""), item.get("original_url") or item.get("image_url", ""),
                             item.get("License", "") or "Public domain / CC0"])
        zf.writestr("credits.csv", "﻿" + credits.getvalue())
    return buf.getvalue(), missing
