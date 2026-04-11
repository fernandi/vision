from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import os
import time

from app.backend.search_engine import VisualSearchEngine

ENV = os.environ.get("ENV", "local")
HF_SOURCE_DATASET = os.environ.get("HF_SOURCE_DATASET", "Mitsua/art-museums-pd-440k")

# Eager startup: load the model + index before accepting requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading search engine...")
    t0 = time.time()
    try:
        search_engine.load()
        print(f"[startup] Ready in {time.time() - t0:.1f}s")
    except Exception as e:
        search_engine.load_error = str(e)
        print(f"[startup] ERROR: {e}")
    yield
    # Nothing to tear down

app = FastAPI(title="Art Visual Search", lifespan=lifespan)

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
    """Return the direct image URL from metadata (already stored during indexing)."""
    if ENV == "production":
        # ImageURL is the original museum URL (IIIF/CDN), stored in index_mapping.json
        image_url = item.get("ImageURL")
        if image_url:
            return image_url
    # Local: served via FastAPI static mount
    return f"/images/{item.get('filename', '')}"


# API Models
class SearchRequest(BaseModel):
    query: str
    page_size: Optional[int] = 20   # results per page
    offset: Optional[int] = 0       # pagination offset within pre-ranked pool
    pool_size: Optional[int] = 200  # total pool to pre-rank (shared across pages)
    diversity: Optional[float] = 0.5

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
        ensure_loaded()  # no-op if already loaded at startup
        t0 = time.time()
        data = search_engine.search(
            req.query,
            pool_size=req.pool_size,
            page_size=req.page_size,
            offset=req.offset,
            diversity=req.diversity,
        )
        elapsed = time.time() - t0
        print(f"[search] '{req.query}' offset={req.offset} → {len(data['results'])} results in {elapsed:.3f}s")
        for item in data["results"]:
            item["image_url"] = get_image_url(item)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ClusterRequest(BaseModel):
    faiss_ids: List[int]

@app.post("/cluster-members")
def cluster_members(req: ClusterRequest):
    """Return full metadata for a list of FAISS IDs (cluster group view)."""
    try:
        ensure_loaded()
        items = search_engine.get_items_by_ids(req.faiss_ids)
        for item in items:
            item["image_url"] = get_image_url(item)
        return {"results": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (local only)
if ENV == "local":
    if os.path.exists("data/images"):
        app.mount("/images", StaticFiles(directory="data/images"), name="images")

# Frontend (root) — always mounted
if os.path.exists("app/frontend"):
    app.mount("/", StaticFiles(directory="app/frontend", html=True), name="frontend")
