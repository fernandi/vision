from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import os
import time
import base64
import io

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
    query: str = ""          # may be empty when searching by image only
    page_size: Optional[int] = 20   # results per page
    offset: Optional[int] = 0       # pagination offset within pre-ranked pool
    pool_size: Optional[int] = 200  # total pool to pre-rank (shared across pages)
    diversity: Optional[float] = 0.5
    reference_image: Optional[str] = None        # single base64 image (legacy)
    reference_images: Optional[List[str]] = None # multiple base64 images
    image_weight: Optional[float] = 0.5          # blend factor: 0=text only, 1=image only
    combination_mode: Optional[str] = "centroid" # how query elements are combined

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

        # Collect all reference images (supports list or legacy single)
        image_embedding = None
        individual_image_embeddings = []   # per-image normalized (1, D) arrays
        b64_list = req.reference_images or ([req.reference_image] if req.reference_image else [])
        if b64_list:
            from PIL import Image
            import torch
            import numpy as np
            processor = search_engine.processor
            model     = search_engine.model
            device    = search_engine.device
            for b64 in b64_list:
                img_data = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.get_image_features(**inputs)
                if not isinstance(output, torch.Tensor):
                    feat = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
                else:
                    feat = output
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                individual_image_embeddings.append(feat.cpu().numpy().astype("float32"))  # (1, D)
            # Averaged embedding (kept for legacy cache key hashing)
            if individual_image_embeddings:
                import numpy as np
                avg  = np.mean(np.stack(individual_image_embeddings, axis=0), axis=0)
                norm = np.linalg.norm(avg, axis=1, keepdims=True)
                image_embedding = (avg / np.where(norm == 0, 1e-9, norm)).astype("float32")

        n_imgs = len(b64_list)
        data = search_engine.search(
            req.query,
            pool_size=req.pool_size,
            page_size=req.page_size,
            offset=req.offset,
            diversity=req.diversity,
            image_embedding=image_embedding,
            image_weight=req.image_weight,
            combination_mode=req.combination_mode or "centroid",
            individual_image_embeddings=individual_image_embeddings or None,
        )
        elapsed = time.time() - t0
        mode = f"text+{n_imgs}img" if n_imgs else "text"
        print(f"[search/{mode}] '{req.query}' offset={req.offset} → {len(data['results'])} results in {elapsed:.3f}s")
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
