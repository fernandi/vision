"""
build_hnsw_index.py
-------------------
Converts the existing IndexFlatIP (brute-force, ~800 MB) into an HNSW32 index.

HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbour
graph.  On 440 k × 512-dim float32 vectors it is typically ~100× faster than
a flat scan with >99 % recall@10.

Usage (run from project root):
    python scripts/build_hnsw_index.py

Output:
    data/faiss_hnsw.index   (~same size as flat index, but queries in <5 ms)

The search_engine.py will automatically prefer faiss_hnsw.index when present.
"""

import os
import time
import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = "data"
FLAT_FILE  = os.path.join(DATA_DIR, "faiss.index")
HNSW_FILE  = os.path.join(DATA_DIR, "faiss_hnsw.index")

# ---------------------------------------------------------------------------
# HNSW hyper-parameters
# ---------------------------------------------------------------------------
# M: number of neighbours per node in the graph.
# Higher M → better recall + larger index + slower construction.
# 32 is the standard sweet-spot for 512-dim embeddings.
HNSW_M = 32

# ef_construction: beam width during graph construction.
# Higher → better recall @ same M, but slower build.
EF_CONSTRUCTION = 200

# ef_search: beam width at query time (can be changed after build).
# 64 gives >99 % recall@10 on typical CLIP embeddings.
EF_SEARCH = 64


def main():
    if not os.path.exists(FLAT_FILE):
        raise FileNotFoundError(f"Flat index not found: {FLAT_FILE}")

    print(f"Loading flat index from {FLAT_FILE} ...")
    t0 = time.time()
    flat_index = faiss.read_index(FLAT_FILE)
    n, d = flat_index.ntotal, flat_index.d
    print(f"  ✓ {n:,} vectors × {d} dims  ({os.path.getsize(FLAT_FILE) // 1_048_576} MB)  [{time.time()-t0:.1f}s]")

    # ------------------------------------------------------------------
    # Reconstruct all raw vectors from the flat index
    # ------------------------------------------------------------------
    print("Reconstructing all vectors ...")
    t0 = time.time()
    vecs = np.zeros((n, d), dtype="float32")
    flat_index.reconstruct_n(0, n, vecs)
    print(f"  ✓ Reconstructed {n:,} vectors [{time.time()-t0:.1f}s]")

    # ------------------------------------------------------------------
    # Build HNSW index
    # IndexHNSWFlat stores the raw vectors (no quantisation) so distances
    # are exact given the approximate graph traversal.
    # For inner-product / cosine similarity we use METRIC_INNER_PRODUCT
    # because the original index is IndexFlatIP (embeddings are L2-normalised).
    # ------------------------------------------------------------------
    print(f"Building HNSW index (M={HNSW_M}, ef_construction={EF_CONSTRUCTION}) ...")
    t0 = time.time()
    hnsw_index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = EF_CONSTRUCTION
    hnsw_index.add(vecs)
    hnsw_index.hnsw.efSearch = EF_SEARCH
    build_time = time.time() - t0
    print(f"  ✓ HNSW built in {build_time:.1f}s")

    # ------------------------------------------------------------------
    # Quick sanity-check: compare top-1 results between flat and HNSW
    # ------------------------------------------------------------------
    print("Sanity-check: comparing flat vs HNSW on 5 random queries ...")
    rng = np.random.default_rng(42)
    sample_ids = rng.integers(0, n, size=5)
    ok = 0
    for sid in sample_ids:
        q = vecs[sid : sid + 1]
        _, flat_ids = flat_index.search(q, 1)
        _, hnsw_ids = hnsw_index.search(q, 1)
        match = flat_ids[0][0] == hnsw_ids[0][0]
        status = "✓" if match else "✗"
        print(f"  {status} id={sid}  flat={flat_ids[0][0]}  hnsw={hnsw_ids[0][0]}")
        if match:
            ok += 1
    print(f"  {ok}/5 top-1 matches")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"Saving HNSW index to {HNSW_FILE} ...")
    t0 = time.time()
    faiss.write_index(hnsw_index, HNSW_FILE)
    size_mb = os.path.getsize(HNSW_FILE) // 1_048_576
    print(f"  ✓ Saved ({size_mb} MB) [{time.time()-t0:.1f}s]")
    print()
    print("Done! Restart the server — it will automatically use faiss_hnsw.index.")


if __name__ == "__main__":
    main()
