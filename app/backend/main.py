from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
from urllib.parse import quote
import os
import time
import base64
import io

import numpy as np
import torch
from PIL import Image

from app.backend import accounts, db, zip_export
from app.backend.search_engine import VisualSearchEngine

ENV = os.environ.get("ENV", "local")
HF_SOURCE_DATASET = os.environ.get("HF_SOURCE_DATASET", "Mitsua/art-museums-pd-440k")

# Eager startup: load the model + index before accepting requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    if not accounts.enabled():
        print("[startup] accounts disabled: set AUTH_SECRET and PUBLIC_BASE_URL")
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

_public_docs = {} if ENV != "production" else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title="Art Visual Search", lifespan=lifespan, **_public_docs)

# Search stays callable from any origin, but never with the session cookie:
# account routes only work from the site itself.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Missing"],
)

# Search Engine Instance
search_engine = VisualSearchEngine()
_load_lock = __import__("threading").Lock()
app.include_router(accounts.router)

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


def _encode_b64_image(b64: str) -> np.ndarray:
    """CLIP-encode an uploaded image → normalised (1, D) float32."""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    inputs = search_engine.processor(images=img, return_tensors="pt").to(search_engine.device)
    with torch.no_grad():
        output = search_engine.model.get_image_features(**inputs)
    if not isinstance(output, torch.Tensor):
        output = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
    feat = output / output.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


def _indexed_embedding(faiss_id: int) -> np.ndarray:
    """Stored vector of an already-indexed image → normalised (1, D) float32.
    Avoids re-downloading the image, which the browser often can't (no CORS)."""
    vec = search_engine.index.reconstruct(int(faiss_id)).reshape(1, -1).astype("float32")
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.where(norm == 0, 1e-9, norm)


# API Models
class SearchRequest(BaseModel):
    query: str = ""          # may be empty when searching by image only
    page_size: Optional[int] = 20   # results per page
    offset: Optional[int] = 0       # pagination offset within pre-ranked pool
    pool_size: Optional[int] = 200  # total pool to pre-rank (shared across pages)
    diversity: Optional[float] = 0.5
    reference_image: Optional[str] = None        # single base64 image (legacy)
    reference_images: Optional[List[str]] = None # multiple base64 images
    image_weight: Optional[float] = 0.5          # blend factor: 0=text only, 1=image only
    combination_mode: Optional[str] = "purified" # how query elements are combined
    negative_images: Optional[List[str]] = None  # base64 negative images
    negative_mode: Optional[str] = "directed"    # directed | orthogonal | penalty
    reference_ids: Optional[List[int]] = None    # indexed images used as references (faiss ids)
    negative_ids: Optional[List[int]] = None     # indexed images used as negatives (faiss ids)

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

        b64_list = req.reference_images or ([req.reference_image] if req.reference_image else [])
        individual_image_embeddings = (
            [_encode_b64_image(b) for b in b64_list]
            + [_indexed_embedding(i) for i in (req.reference_ids or [])]
        )
        image_embedding = None   # averaged, used for the server-side cache key
        if individual_image_embeddings:
            avg  = np.mean(np.stack(individual_image_embeddings, axis=0), axis=0)
            norm = np.linalg.norm(avg, axis=1, keepdims=True)
            image_embedding = (avg / np.where(norm == 0, 1e-9, norm)).astype("float32")

        negative_embeddings = (
            [_encode_b64_image(b) for b in (req.negative_images or [])]
            + [_indexed_embedding(i) for i in (req.negative_ids or [])]
        )

        n_imgs = len(individual_image_embeddings)
        data = search_engine.search(
            req.query,
            pool_size=req.pool_size,
            page_size=req.page_size,
            offset=req.offset,
            diversity=req.diversity,
            image_embedding=image_embedding,
            image_weight=req.image_weight,
            combination_mode=req.combination_mode or "purified",
            individual_image_embeddings=individual_image_embeddings or None,
            negative_embeddings=negative_embeddings or None,
            negative_mode=req.negative_mode or "directed",
        )
        elapsed = time.time() - t0
        mode = f"text+{n_imgs}img" if n_imgs else "text"
        print(f"[search/{mode}] '{req.query}' offset={req.offset} → {len(data['results'])} results in {elapsed:.3f}s")
        for item in data["results"]:
            item["image_url"] = get_image_url(item)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ZipRequest(BaseModel):
    name: str = "collection"
    faiss_ids: List[int]


@app.post("/collection-zip")
def collection_zip(req: ZipRequest):
    """Zip a collection's images. URLs come from the index, never from the client."""
    ensure_loaded()
    items = search_engine.get_items_by_ids(req.faiss_ids[:zip_export.MAX_ITEMS])
    for item in items:
        item["image_url"] = get_image_url(item)
    data, missing = zip_export.build_zip(items)
    filename = f"{zip_export.safe_name(req.name)}.zip"
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
        "X-Missing": str(missing),
    })


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
