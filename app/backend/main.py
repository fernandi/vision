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

@app.on_event("startup")
async def startup_event():
    try:
        search_engine.load()
    except Exception as e:
        print(f"Error loading search engine: {e}")


def get_image_url(item: dict) -> str:
    """Return the correct image URL depending on environment."""
    if ENV == "production":
        # Use the HF dataset index if stored, fallback to our sequential id
        hf_idx = item.get("hf_idx", item.get("id", 0))
        return f"/image/{hf_idx}"
    # Local: served via FastAPI static mount
    return f"/images/{item.get('filename', '')}"


# ── Image proxy endpoint (production) ────────────────────────────────────────
@app.get("/image/{hf_idx}")
async def proxy_image(hf_idx: int):
    """Fetch image URL from HF dataset server and redirect to it."""
    if ENV == "local":
        raise HTTPException(status_code=404, detail="Use /images/ static mount in local mode")

    ds_server_url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={HF_SOURCE_DATASET}&config=default&split=train"
        f"&offset={hf_idx}&length=1"
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(ds_server_url)
            resp.raise_for_status()
            data = resp.json()
            image_src = data["rows"][0]["row"]["jpg"]["src"]
            return RedirectResponse(url=image_src, status_code=302)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch image from HF: {e}")


# API Models
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20

# Routes
@app.get("/health")
def health_check():
    return {"status": "ok", "env": ENV, "indexed": len(search_engine.metadata_mapping) or search_engine.index.ntotal if search_engine.index else 0}

@app.post("/search")
def search(req: SearchRequest):
    try:
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
