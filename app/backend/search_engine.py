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
        self.hnsw_file  = os.path.join(data_dir, "faiss_hnsw.index")  # preferred if present
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

        print(f"Loading FAISS index...")
        # Prefer HNSW (fast ANN) over flat index (brute-force)
        chosen_index_file = (
            self.hnsw_file if os.path.exists(self.hnsw_file) else self.index_file
        )
        index_label = "HNSW" if os.path.exists(self.hnsw_file) else "Flat"
        if not os.path.exists(chosen_index_file):
            raise FileNotFoundError(f"Index not found at {chosen_index_file}. Run index_data.py first.")

        self.index = faiss.read_index(chosen_index_file)

        # Tune efSearch for HNSW indexes (no-op for flat indexes)
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = 64
            print(f"  ✓ {index_label} index loaded ({self.index.ntotal:,} vectors, efSearch=64)")
        else:
            print(f"  ✓ {index_label} index loaded ({self.index.ntotal:,} vectors)")


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
        Maximal Marginal Relevance re-ranking — fully vectorised.

        Uses numpy matrix operations instead of a Python loop over candidates,
        giving ~10-30x speedup on large candidate sets.

        diversity=0.0 → pure relevance (FAISS order)
        diversity=1.0 → pure diversity
        diversity=0.5 → recommended default

        Returns:
            (selected_ids, cluster_sizes) where cluster_sizes is a dict mapping
            each selected faiss_id to its Voronoi cluster size: the number of
            candidates (from the full fetch_k pool) that are nearest to it.
            This tells you how many images were "represented" / suppressed by
            each selected result.
        """
        if not candidate_ids:
            return [], {}

        n = len(candidate_ids)
        k = min(k, n)

        # Retrieve raw embeddings from FAISS for all candidates in one batch
        vecs = np.vstack([
            self.index.reconstruct(int(fid)) for fid in candidate_ids
        ]).astype('float32')  # shape (N, D)

        # L2-normalize so dot product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        vecs = vecs / norms  # (N, D), unit vectors

        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)  # (D,)
        query_sims = (vecs @ query_norm).flatten()  # (N,) — relevance scores

        # Precompute the full similarity matrix between all candidates
        # inter_sim[i,j] = cosine similarity between candidate i and candidate j
        inter_sim = vecs @ vecs.T  # (N, N)

        selected = []          # indices into candidate_ids (ordered)
        mask = np.ones(n, dtype=bool)  # True = still available

        for step in range(k):
            if step == 0:
                # First pick: pure relevance
                available = np.where(mask)[0]
                chosen = available[int(np.argmax(query_sims[available]))]
            else:
                # Vectorised MMR: for each remaining candidate compute
                #   mmr = (1-λ)·relevance − λ·max_sim_to_selected
                # max_sim_to_selected is a column-max over selected rows
                # inter_sim[selected, :] has shape (S, N); max over axis=0 → (N,)
                available = np.where(mask)[0]
                redundancy = inter_sim[np.ix_(selected, available)].max(axis=0)  # (n_avail,)
                mmr_scores = (
                    (1 - diversity) * query_sims[available]
                    - diversity * redundancy
                )
                chosen = available[int(np.argmax(mmr_scores))]

            selected.append(chosen)
            mask[chosen] = False

        selected_ids = [candidate_ids[i] for i in selected]

        # ── Pass 1 : Union-Find deduplication on MMR-selected pool ──────────────
        # inter_sim[i,j] is already computed above. Use it to merge near-duplicates
        # that slipped through MMR (can happen at lower λ values).
        # dup_threshold=0.91: calibrated against real data — catches same artwork in
        # different reproductions (sim ≈ 0.91-0.95) without merging merely "similar"
        # scenes (which sit at ≈ 0.84-0.88).
        dup_threshold = 0.91
        n_sel = len(selected)
        # Sub-matrix: similarity among MMR-selected items (S × S)
        sim_sel = inter_sim[np.ix_(selected, selected)]  # (S, S)

        parent = list(range(n_sel))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(i, j):
            pi, pj = _find(i), _find(j)
            if pi == pj:
                return
            si = candidate_scores.get(selected_ids[pi], 0.0)
            sj = candidate_scores.get(selected_ids[pj], 0.0)
            if si >= sj:
                parent[pj] = pi
            else:
                parent[pi] = pj

        above = np.argwhere((sim_sel > dup_threshold) & ~np.eye(n_sel, dtype=bool))
        for _i, _j in above:
            if _i < _j:
                _union(int(_i), int(_j))

        seen_reps, deduped_local, deduped_ids = set(), [], []
        for i in range(n_sel):
            rep = _find(i)
            if rep not in seen_reps:
                seen_reps.add(rep)
                deduped_local.append(selected[rep])  # index into candidate_ids
                deduped_ids.append(selected_ids[rep])

        # ── Pass 2 : Voronoi partition of ALL N candidates onto surviving reps ──
        ded_arr      = np.array(deduped_local)         # (D,)
        ded_sim      = inter_sim[ded_arr, :]           # (D, N)
        nearest_pos  = ded_sim.argmax(axis=0)          # (N,) index into deduped_ids

        cluster_sizes   = {}
        cluster_members = {sid: [] for sid in deduped_ids}
        for cand_idx, rep_pos in enumerate(nearest_pos.tolist()):
            sid  = deduped_ids[rep_pos]
            cfid = candidate_ids[cand_idx]
            cluster_sizes[sid]   = cluster_sizes.get(sid, 0) + 1
            cluster_members[sid].append(cfid)

        return deduped_ids, cluster_sizes, cluster_members

    def _deduplicate_and_voronoi(self, selected_ids, all_candidate_ids, score_by_id,
                                  dup_threshold=0.91):
        """
        Two-pass clustering that fixes near-duplicates appearing in the displayed pool.

        Pass 1 — Union-Find deduplication within the selected pool:
            Merges pairs of selected images whose cosine similarity exceeds
            `dup_threshold` (captures same artwork in different reproductions,
            near-identical prints, etc.). The highest-scoring member of each
            group becomes the representative; others become cluster members.

        Pass 2 — Voronoi partition:
            Assigns every candidate in `all_candidate_ids` (the full over-fetched
            pool) to the nearest surviving representative.  This combines the
            over-fetch non-selected candidates AND the merged duplicates into
            cohesive clusters.

        Args:
            selected_ids:       top-N FAISS ids (may contain visual duplicates).
            all_candidate_ids:  full 3× over-fetch pool (superset of selected_ids).
            score_by_id:        {faiss_id: cosine_score} dict.
            dup_threshold:      cosine similarity above which two selected images
                                are considered near-duplicates and merged.
        Returns:
            (deduped_ids, cluster_sizes, cluster_members)
        """
        n_sel = len(selected_ids)
        if n_sel == 0:
            return [], {}, {}

        # ── Batch-reconstruct & normalise all embeddings (one shot) ──────────────
        all_vecs = np.vstack([
            self.index.reconstruct(int(fid)) for fid in all_candidate_ids
        ]).astype('float32')                        # (M, D)
        norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        all_vecs /= np.where(norms == 0, 1e-9, norms)

        id_to_local = {fid: i for i, fid in enumerate(all_candidate_ids)}
        sel_indices = np.array([id_to_local[fid] for fid in selected_ids])  # (S,)
        sel_vecs    = all_vecs[sel_indices]         # (S, D)

        # ── Pass 1: Union-Find deduplication within selected pool ──────────────
        sim_sel = sel_vecs @ sel_vecs.T             # (S, S) pairwise similarity

        parent = list(range(n_sel))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]       # path halving
                x = parent[x]
            return x

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi == pj:
                return
            # Higher-score item becomes the root (representative)
            si = score_by_id.get(selected_ids[pi], 0.0)
            sj = score_by_id.get(selected_ids[pj], 0.0)
            if si >= sj:
                parent[pj] = pi
            else:
                parent[pi] = pj

        # Iterate over all pairs above threshold
        above = np.argwhere(
            (sim_sel > dup_threshold) & ~np.eye(n_sel, dtype=bool)
        )
        for i, j in above:
            if i < j:
                union(int(i), int(j))

        # Surviving representatives in original FAISS score order
        seen, deduped_ids = set(), []
        for i in range(n_sel):
            rep = find(i)
            if rep not in seen:
                seen.add(rep)
                deduped_ids.append(selected_ids[rep])

        # ── Pass 2: Voronoi partition of ALL candidates onto survivors ──────────
        ded_local   = [id_to_local[fid] for fid in deduped_ids]
        ded_vecs    = all_vecs[ded_local]           # (D, dim)
        cross_sim   = ded_vecs @ all_vecs.T         # (D, M)
        nearest_pos = cross_sim.argmax(axis=0)      # (M,) index into deduped_ids

        cluster_sizes   = {}
        cluster_members = {sid: [] for sid in deduped_ids}
        for cand_idx, rep_pos in enumerate(nearest_pos.tolist()):
            sid  = deduped_ids[rep_pos]
            cfid = all_candidate_ids[cand_idx]
            cluster_sizes[sid]   = cluster_sizes.get(sid, 0) + 1
            cluster_members[sid].append(cfid)

        return deduped_ids, cluster_sizes, cluster_members

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

            # --- FAISS search: over-fetch for both MMR headroom and Voronoi clustering ---
            # Always fetch 3× candidates so the no-diversity mode can also compute
            # Voronoi cluster sizes (near-duplicates not shown in the top-N).
            fetch_k = min(pool_size * 3, self.index.ntotal)
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

            # --- Rank / re-rank to pool_size ---
            cluster_sizes   = {}   # faiss_id → Voronoi count
            cluster_members = {}   # faiss_id → [faiss_ids in cluster]

            if diversity > 0.0 and len(valid_ids) > pool_size:
                # MMR: selects diverse pool, then dedup+Voronoi inside (one shot, zero
                # extra reconstruction — inter_sim is already computed during MMR)
                selected_ids, cluster_sizes, cluster_members = self._mmr_rerank(
                    text_embedding[0], valid_ids, score_by_id, pool_size, diversity
                )
            else:
                # Pure relevance: top-N by FAISS score
                selected_ids = valid_ids[:pool_size]
                # Dedup within pool + Voronoi on over-fetched candidates
                if len(valid_ids) > pool_size:
                    selected_ids, cluster_sizes, cluster_members = \
                        self._deduplicate_and_voronoi(selected_ids, valid_ids, score_by_id)

            # --- Fetch all metadata for the pool ---
            meta_by_id = self._lookup_metadata(selected_ids)
            pool = []
            for fid in selected_ids:
                if fid not in meta_by_id:
                    continue
                item = dict(meta_by_id[fid])
                item['score'] = score_by_id.get(fid, 0.0)
                if cluster_sizes:
                    item['cluster_size']       = cluster_sizes.get(fid, 1)
                    # Sort members by FAISS score (descending) so most
                    # relevant image in each cluster comes first
                    members = cluster_members.get(fid, [fid])
                    members_sorted = sorted(members,
                                           key=lambda x: score_by_id.get(x, 0.0),
                                           reverse=True)
                    item['cluster_member_ids'] = members_sorted
                pool.append(item)

            self._pool_cache_set(cache_key, pool)

        # --- Slice the pool for this page ---
        page = pool[offset: offset + page_size]
        return {
            "results": page,
            "total": len(pool),
            "has_more": (offset + page_size) < len(pool),
        }

    def get_items_by_ids(self, faiss_ids):
        """
        Return metadata for a list of FAISS IDs (used by /cluster-members endpoint).
        Preserves the order of faiss_ids.
        """
        meta_by_id = self._lookup_metadata(faiss_ids)
        result = []
        for fid in faiss_ids:
            if fid in meta_by_id:
                result.append(dict(meta_by_id[fid]))
        return result
