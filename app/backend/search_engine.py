import torch
import faiss
import numpy as np
import json
import os
import sqlite3
from transformers import CLIPProcessor, CLIPModel

class VisualSearchEngine:
    def __init__(self, data_dir="data", model_id="openai/clip-vit-base-patch32"):
        self.data_dir = data_dir
        self.index_file = os.path.join(data_dir, "faiss.index")
        self.mapping_file = os.path.join(data_dir, "index_mapping.json")
        self.db_file = os.path.join(data_dir, "metadata.db")
        self.model_id = model_id

        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

        print(f"Stats: Using device {self.device}")

        self.model = None
        self.processor = None
        self.index = None
        self.db_conn = None
        self.db_columns = []
        self.denylist: set = set()  # faiss_ids to always suppress from results
        self.metadata_mapping = []
        self.load_error = None  # exposed via /health for debugging

        # LRU cache for pre-ranked result pools (avoids re-running CLIP+MMR on scroll)
        self._pool_cache: dict = {}   # key → list of metadata dicts (full pool)
        self._pool_cache_order: list = []  # insertion order for LRU eviction
        self._pool_cache_max = 10     # keep last N unique queries


    def _download_from_hf(self):
        """Download FAISS index and metadata from HuggingFace Hub (production mode)."""
        from huggingface_hub import hf_hub_download
        hf_repo = os.environ.get("HF_INDEX_REPO")
        hf_token = os.environ.get("HF_TOKEN")

        if not hf_repo:
            raise ValueError("HF_INDEX_REPO environment variable not set.")

        print(f"Downloading index from HuggingFace Hub: {hf_repo} ...")
        os.makedirs(self.data_dir, exist_ok=True)

        self.index_file = hf_hub_download(
            repo_id=hf_repo, filename="faiss.index",
            repo_type="dataset", token=hf_token, local_dir=self.data_dir,
        )
        print("  ✓ faiss.index")

        # Prefer SQLite db if available
        try:
            self.db_file = hf_hub_download(
                repo_id=hf_repo, filename="metadata.db",
                repo_type="dataset", token=hf_token, local_dir=self.data_dir,
            )
            print("  ✓ metadata.db")
        except Exception:
            # Fall back to JSON
            self.mapping_file = hf_hub_download(
                repo_id=hf_repo, filename="index_mapping.json",
                repo_type="dataset", token=hf_token, local_dir=self.data_dir,
            )
            print("  ✓ index_mapping.json (fallback)")

    def _load_metadata(self):
        """Load metadata from SQLite if available, else fall back to JSON."""
        if os.path.exists(self.db_file):
            print(f"Loading metadata from SQLite: {self.db_file}")
            self.db_conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.db_conn.row_factory = sqlite3.Row
            cursor = self.db_conn.execute("SELECT name FROM pragma_table_info('images')")
            self.db_columns = [row[0] for row in cursor.fetchall()]
            n = self.db_conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            print(f"  ✓ SQLite loaded ({n} rows, ~{os.path.getsize(self.db_file)//1024//1024} MB on disk)")
        else:
            print(f"Loading metadata from JSON: {self.mapping_file}")
            with open(self.mapping_file, "r") as f:
                self.metadata_mapping = json.load(f)
            print(f"  ✓ JSON loaded ({len(self.metadata_mapping)} items)")

    def load(self):
        env = os.environ.get("ENV", "local")

        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

        # Apply dynamic int8 quantization for ~2x speedup on CPU text encoding.
        # Skip on MPS/CUDA (GPU is fast, quantization not supported there anyway).
        if self.device == "cpu":
            try:
                import platform
                backend = "fbgemm" if platform.machine() in ("x86_64", "AMD64") else "qnnpack"
                torch.backends.quantized.engine = backend
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print(f"  ✓ int8 quantization applied (backend={backend})")
            except Exception as e:
                print(f"  ⚠ int8 quantization failed, using float32: {e}")
        else:
            print(f"  ↷ int8 skipped (device={self.device})")

        if env == "production":
            self._download_from_hf()

        print(f"Loading FAISS index from {self.index_file}...")
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Index not found at {self.index_file}. Run index_data.py first.")

        # Regular read (mmap can cause issues in some container environments)
        self.index = faiss.read_index(self.index_file)


        self._load_metadata()
        self._build_denylist()

        n_indexed = self.index.ntotal
        print(f"Search Engine Ready. ({n_indexed} images indexed)")

    def _build_denylist(self):
        """
        Build a set of faiss_ids to always suppress from search results.

        Two parasite categories are excluded:
        1. MET DVB ceramic shards — small terracotta/pottery fragments photographed
           with a ruler and color-calibration card (ImageURL contains 'DVB').
           CLIP's embedding is dominated by the ruler background, not the artifact.
        2. Smithsonian National Postal Museum stamp plate proofs — large grids of
           identical stamps that create 'hub' embeddings (high avg. similarity to
           everything due to their repetitive, neutral-background structure).
        """
        if self.db_conn is None:
            print("  ⚠ denylist: no SQLite connection, skipping.")
            return

        dvb_ids = {
            row[0] for row in self.db_conn.execute(
                "SELECT faiss_id FROM images WHERE ImageURL LIKE '%/DVB%'"
            )
        }
        npm_ids = {
            row[0] for row in self.db_conn.execute(
                "SELECT faiss_id FROM images WHERE ImageID LIKE 'NPM-%'"
            )
        }
        self.denylist = dvb_ids | npm_ids
        print(f"  ✓ denylist built: {len(dvb_ids)} DVB ceramic shards + "
              f"{len(npm_ids)} NPM stamp sheets = {len(self.denylist)} total suppressed")

    def _lookup_metadata(self, faiss_ids):
        """Return list of metadata dicts for given FAISS index IDs."""
        if self.db_conn is not None:
            ids = [int(i) for i in faiss_ids if i >= 0]
            placeholders = ",".join("?" * len(ids))
            cursor = self.db_conn.execute(
                f"SELECT * FROM images WHERE faiss_id IN ({placeholders})", ids
            )
            rows = {row["faiss_id"]: dict(row) for row in cursor.fetchall()}
            return rows
        else:
            # Fallback: list lookup
            return {i: dict(self.metadata_mapping[i])
                    for i in faiss_ids if 0 <= i < len(self.metadata_mapping)}

    def _mmr_rerank(self, query_vec, candidate_ids, candidate_scores, k, diversity):
        """
        Maximal Marginal Relevance re-ranking.

        Iteratively selects k items from candidates that balance:
        - relevance: cosine similarity to the query vector
        - diversity: dissimilarity to already-selected items

        diversity=0.0 → pure relevance (FAISS order)
        diversity=1.0 → pure diversity
        diversity=0.7 → recommended default
        """
        if not candidate_ids:
            return []

        # Retrieve raw embeddings from FAISS for all candidates
        vecs = np.vstack([
            self.index.reconstruct(int(fid)) for fid in candidate_ids
        ]).astype('float32')

        # L2-normalize so dot product = cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        vecs = vecs / norms

        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        # Relevance scores: cosine similarity to query (already normalised)
        query_sims = (vecs @ query_norm.T).flatten()

        selected_indices = []   # positions within candidate_ids list
        remaining = list(range(len(candidate_ids)))

        for _ in range(min(k, len(candidate_ids))):
            if not remaining:
                break

            if not selected_indices:
                # First pick: highest relevance
                best_pos = int(np.argmax([query_sims[i] for i in remaining]))
                chosen = remaining[best_pos]
            else:
                # MMR score for each remaining candidate
                selected_vecs = vecs[selected_indices]  # shape (S, D)
                best_score = -np.inf
                chosen = remaining[0]

                for i in remaining:
                    relevance = query_sims[i]
                    # Maximum cosine sim to any already-selected item
                    redundancy = float(np.max(vecs[i] @ selected_vecs.T))
                    mmr = (1 - diversity) * relevance - diversity * redundancy
                    if mmr > best_score:
                        best_score = mmr
                        chosen = i

            selected_indices.append(chosen)
            remaining.remove(chosen)

        return [candidate_ids[i] for i in selected_indices]

    def _pool_cache_get(self, key):
        if key in self._pool_cache:
            # Move to end (most recently used)
            self._pool_cache_order.remove(key)
            self._pool_cache_order.append(key)
            return self._pool_cache[key]
        return None

    def _pool_cache_set(self, key, pool):
        if key in self._pool_cache:
            self._pool_cache_order.remove(key)
        elif len(self._pool_cache_order) >= self._pool_cache_max:
            oldest = self._pool_cache_order.pop(0)
            del self._pool_cache[oldest]
        self._pool_cache[key] = pool
        self._pool_cache_order.append(key)

    def search(self, query_text, pool_size=200, page_size=20, offset=0, diversity=0.5):
        """
        Paginated search with MMR diversity.

        On the first call for a (query, diversity) pair, computes a full ranked
        pool of up to `pool_size` results (CLIP encode + FAISS + MMR) and caches
        it server-side. Subsequent offset calls just slice the cached pool —
        no re-encoding or re-ranking.

        Args:
            query_text:  natural language query
            pool_size:   total diverse results to pre-rank (default 200)
            page_size:   results to return in this response (default 20)
            offset:      starting index within the ranked pool (default 0)
            diversity:   MMR lambda (0=relevance-only, 1=diversity-only)
        Returns:
            dict with keys: results (list), total (int), has_more (bool)
        """
        if not self.model:
            self.load()

        cache_key = (query_text.strip().lower(), round(diversity, 2))
        pool = self._pool_cache_get(cache_key)

        if pool is None:
            # --- Encode query ---
            inputs = self.processor(text=[query_text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                if not isinstance(outputs, torch.Tensor):
                    text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
                else:
                    text_features = outputs

            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy().astype('float32')

            # --- FAISS search: over-fetch for MMR room ---
            # Fetch 5× pool_size candidates (capped at index size)
            fetch_k = pool_size if diversity <= 0.0 else min(pool_size * 5, self.index.ntotal)
            distances, indices = self.index.search(text_embedding, fetch_k)

            valid_ids = [
                int(idx) for idx in indices[0]
                if idx >= 0 and int(idx) not in self.denylist
            ]
            score_by_id = {
                int(idx): float(distances[0][i])
                for i, idx in enumerate(indices[0])
                if idx >= 0 and int(idx) not in self.denylist
            }

            # --- MMR re-rank to pool_size ---
            if diversity > 0.0 and len(valid_ids) > pool_size:
                selected_ids = self._mmr_rerank(
                    text_embedding[0], valid_ids, score_by_id, pool_size, diversity
                )
            else:
                selected_ids = valid_ids[:pool_size]

            # --- Fetch all metadata for the pool ---
            meta_by_id = self._lookup_metadata(selected_ids)
            pool = []
            for fid in selected_ids:
                if fid not in meta_by_id:
                    continue
                item = dict(meta_by_id[fid])
                item['score'] = score_by_id.get(fid, 0.0)
                pool.append(item)

            self._pool_cache_set(cache_key, pool)

        # --- Slice the pool for this page ---
        page = pool[offset: offset + page_size]
        return {
            "results": page,
            "total": len(pool),
            "has_more": (offset + page_size) < len(pool),
        }
