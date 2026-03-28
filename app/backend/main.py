from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx

from app.backend.search_engine import VisualSearchEngine

app = FastAPI(title="Art Visual Search")

ENV = os.environ.get("ENV", "local")
HF_SOURCE_DATASET = os.environ.get("HF_SOURCE_DATASET", "Mitsua/art-museums-pd-440k")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Search Engine Instance
search_engine = VisualSearchEngine()
_load_lock = __import__("threading").Lock()

def ensure_loaded():
    """Load the search engine lazily on first request."""
    if search_engine.index is not None:
        return
    with _load_lock:
        if search_engine.index is not None:
            return
        try:
            search_engine.load()
        except Exception as e:
            search_engine.load_error = str(e)
            print(f"ERROR loading search engine: {e}")
            raise


def get_image_url(item: dict) -> str:
    """Return the correct image URL depending on environment."""
    if ENV == "production":
        # Use the HF dataset index if stored, fallback to our sequential id
        hf_idx = item.get("hf_idx", item.get("id", 0))
        return f"/image/{hf_idx}"
    # Local: served via FastAPI static mount
    return f"/images/{item.get('filename', '')}"


# In-memory cache: hf_idx → signed image URL
_image_url_cache: dict = {}

async def _fetch_hf_image_url(hf_idx: int) -> str:
    """Fetch image URL from HF Dataset Server with retry + cache."""
    if hf_idx in _image_url_cache:
        return _image_url_cache[hf_idx]

    ds_server_url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={HF_SOURCE_DATASET}&config=default&split=train"
        f"&offset={hf_idx}&length=1"
    )
    last_err = None
    for attempt in range(4):  # up to 4 attempts
        if attempt > 0:
            await __import__("asyncio").sleep(2 ** (attempt - 1))  # 1s, 2s, 4s
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(ds_server_url)
                resp.raise_for_status()
                data = resp.json()
                # Try common column names for image
                row = data["rows"][0]["row"]
                img = row.get("jpg") or row.get("image") or next(
                    (v for v in row.values() if isinstance(v, dict) and "src" in v), None
                )
                if img is None:
                    raise ValueError(f"No image field found in row. Keys: {list(row.keys())}")
                image_url = img["src"]
                _image_url_cache[hf_idx] = image_url  # cache for future requests
                return image_url
        except Exception as e:
            last_err = e

    raise last_err


# ── Image proxy endpoint (production) ────────────────────────────────────────
@app.get("/image/{hf_idx}")
async def proxy_image(hf_idx: int):
    """Redirect to HF-hosted image URL (cached + retried)."""
    if ENV == "local":
        raise HTTPException(status_code=404, detail="Use /images/ static mount in local mode")
    try:
        image_url = await _fetch_hf_image_url(hf_idx)
        return RedirectResponse(url=image_url, status_code=302)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image from HF: {e}")


# API Models
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20

# Routes
@app.get("/health")
def health_check():
    n = 0
    if search_engine.index:
        n = search_engine.index.ntotal
    elif search_engine.metadata_mapping:
        n = len(search_engine.metadata_mapping)
    return {
        "status": "ok",
        "env": ENV,
        "indexed": n,
        "load_error": search_engine.load_error,
    }

@app.post("/search")
def search(req: SearchRequest):
    try:
        ensure_loaded()
        results = search_engine.search(req.query, k=req.limit)
        for item in results:
            item["image_url"] = get_image_url(item)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (local only)
if ENV == "local":
    if os.path.exists("data/images"):
        app.mount("/images", StaticFiles(directory="data/images"), name="images")

# Frontend (root) — always mounted
if os.path.exists("app/frontend"):
    app.mount("/", StaticFiles(directory="app/frontend", html=True), name="frontend")
