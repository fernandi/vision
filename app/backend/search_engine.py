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
        self.corpus_mean = None  # unit-norm mean of a random sample of indexed vectors

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
        self._compute_corpus_mean()

        n_indexed = self.index.ntotal
        print(f"Search Engine Ready. ({n_indexed} images indexed)")

    def _compute_corpus_mean(self, n_sample: int = 500):
        """
        Estimate the corpus mean by sampling n_sample random vectors from the index.
        Used by the 'purified' combination mode to subtract the 'generic art' direction.
        """
        if self.index is None or self.index.ntotal == 0:
            return
        n_total   = self.index.ntotal
        n_sample  = min(n_sample, n_total)
        rng       = np.random.default_rng(42)

        # Try batch reconstruction (works for IndexFlat* and most stored-vector indices).
        # Fall back to the individual-call loop for index types that don't support it.
        try:
            # reconstruct_n(start, n) is a single C++ call — much faster than a loop.
            # We pick a random contiguous block then subsample from it.
            start = int(rng.integers(0, max(1, n_total - n_sample)))
            sample_vecs = np.zeros((n_sample, self.index.d), dtype='float32')
            self.index.reconstruct_n(start, n_sample, sample_vecs)
        except Exception:
            # Fallback: individual reconstruct calls
            sample_ids  = rng.choice(n_total, n_sample, replace=False)
            sample_vecs = np.vstack(
                [self.index.reconstruct(int(i)) for i in sample_ids]
            ).astype('float32')

        norms = np.linalg.norm(sample_vecs, axis=1, keepdims=True)
        sample_vecs /= np.where(norms == 0, 1e-9, norms)
        cm      = sample_vecs.mean(axis=0)
        norm_cm = np.linalg.norm(cm)
        self.corpus_mean = (cm / (norm_cm + 1e-9)).astype('float32')
        print(f"  ✓ corpus mean estimated (n={n_sample})")

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

        # ── Cap cluster sizes (promotes overflow to new gallery cards) ────────────
        id_to_local = {fid: i for i, fid in enumerate(candidate_ids)}
        deduped_ids, cluster_sizes, cluster_members = self._cap_cluster_sizes(
            deduped_ids, vecs, id_to_local, cluster_sizes, cluster_members, candidate_scores
        )

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

        # ── Cap cluster sizes (promotes overflow to new gallery cards) ────────────
        deduped_ids, cluster_sizes, cluster_members = self._cap_cluster_sizes(
            deduped_ids, all_vecs, id_to_local, cluster_sizes, cluster_members, score_by_id
        )

        return deduped_ids, cluster_sizes, cluster_members

    def _cap_cluster_sizes(
        self, deduped_ids, all_vecs, id_to_local,
        cluster_sizes, cluster_members, score_by_id,
        max_size=15, very_high_threshold=0.97
    ):
        """
        Ensure no Voronoi cluster has more than `max_size` members by
        recursively promoting overflow members as new gallery centroids.

        Exception: members whose cosine similarity to the centroid exceeds
        `very_high_threshold` (true duplicates — exact scans/copies of the
        same artwork) are never counted against the cap and always stay
        grouped together.

        Promoted centroids are appended to deduped_ids so they appear as
        additional gallery cards; no images are silently discarded.

        Args:
            deduped_ids:         current list of centroid faiss_ids.
            all_vecs:            normalised embedding matrix (M, D) for the
                                 full candidate pool. Must include rows for
                                 every id in cluster_members values.
            id_to_local:         {faiss_id: row_index_in_all_vecs}.
            cluster_sizes:       {centroid_id: int}.
            cluster_members:     {centroid_id: [faiss_id, ...]}.
            score_by_id:         {faiss_id: float} FAISS cosine score.
            max_size:            maximum cluster members (default 15).
            very_high_threshold: sim above which a member is a 'true copy'
                                 and ignored by the cap (default 0.97).
        Returns:
            (deduped_ids, cluster_sizes, cluster_members) with all clusters
            satisfying len(members) ≤ max_size (or all members are true copies).
        """
        result_ids    = list(deduped_ids)
        result_sizes  = dict(cluster_sizes)
        result_members = {sid: list(ms) for sid, ms in cluster_members.items()}

        to_check = list(deduped_ids)   # queue of centroids to inspect

        while to_check:
            sid = to_check.pop(0)
            members = result_members.get(sid, [])

            if len(members) <= max_size:
                continue

            # Similarity of each member to centroid
            c_vec      = all_vecs[id_to_local[sid]]
            m_vecs     = np.array([all_vecs[id_to_local[m]] for m in members])  # (K, D)
            m_sims     = (m_vecs @ c_vec).tolist()                               # (K,)

            very_high  = [(m, s) for m, s in zip(members, m_sims)
                          if s > very_high_threshold]
            regular    = [(m, s) for m, s in zip(members, m_sims)
                          if s <= very_high_threshold]

            cap = max(0, max_size - len(very_high))
            if len(regular) <= cap:
                # Nothing to do (all overflow is very-high-sim)
                result_members[sid] = [m for m, _ in very_high] + [m for m, _ in regular]
                result_sizes[sid]   = len(result_members[sid])
                continue

            # Sort regular by similarity to centroid: keep the most similar ones
            regular_sorted  = sorted(regular, key=lambda x: x[1], reverse=True)
            kept            = regular_sorted[:cap]
            overflow        = regular_sorted[cap:]   # excess — need new centroid

            # Update original cluster
            result_members[sid] = [m for m, _ in very_high] + [m for m, _ in kept]
            result_sizes[sid]   = len(result_members[sid])

            # Promote the highest-scoring (by FAISS relevance) overflow member
            overflow_by_score   = sorted(overflow,
                                         key=lambda x: score_by_id.get(x[0], 0.0),
                                         reverse=True)
            new_cid             = overflow_by_score[0][0]
            remaining_overflow  = [m for m, _ in overflow_by_score[1:]]

            # Assign remaining overflow to whichever centroid (original / new) is closer
            new_c_vec  = all_vecs[id_to_local[new_cid]]
            new_members = [new_cid]
            for rm in remaining_overflow:
                rv = all_vecs[id_to_local[rm]]
                if float(rv @ new_c_vec) >= float(rv @ c_vec):
                    new_members.append(rm)
                else:
                    result_members[sid].append(rm)
                    result_sizes[sid] += 1

            result_members[new_cid] = new_members
            result_sizes[new_cid]   = len(new_members)

            if new_cid not in result_ids:
                result_ids.append(new_cid)

            # Re-check both if still too large
            if result_sizes[sid] > max_size:
                to_check.append(sid)
            if result_sizes[new_cid] > max_size:
                to_check.append(new_cid)

        return result_ids, result_sizes, result_members

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

    def search(self, query_text, pool_size=200, page_size=20, offset=0, diversity=0.5,
                image_embedding=None, image_weight=0.5,
                combination_mode="purified", individual_image_embeddings=None,
                negative_embeddings=None, negative_mode="directed"):
        """
        Paginated search with MMR diversity.

        On the first call for a (query, diversity, combination_mode) triple,
        computes a full ranked pool of up to `pool_size` results and caches it
        server-side. Subsequent offset calls just slice the cached pool.

        Args:
            query_text:                  natural language query (may be empty)
            pool_size:                   total diverse results to pre-rank (default 200)
            page_size:                   results per response (default 20)
            offset:                      pagination offset (default 0)
            diversity:                   MMR lambda (0=relevance-only, 1=diversity-only)
            image_embedding:             (1, D) float32 — averaged image embedding (legacy)
            image_weight:                text/image blend for centroid mode (default 0.5)
            combination_mode:            how query elements are combined:
                'centroid'   — weighted average (default, current behaviour)
                'intersection'— min-similarity across elements (AND logic)
                'union'      — max-similarity across elements (OR logic)
                'pca'        — first principal component of query matrix
                'purified'   — query mean minus corpus mean (amplifies specific attributes)
            individual_image_embeddings: list of (1, D) float32 arrays, one per image.
                                         When provided, used instead of image_embedding.
        Returns:
            dict with keys: results (list), total (int), has_more (bool)
        """
        if not self.model:
            self.load()

        # --- Cache key ---
        img_hash = None
        if image_embedding is not None:
            img_hash = hash(image_embedding.tobytes())
        neg_hash = None
        if negative_embeddings:
            neg_hash = hash(tuple(e.tobytes() for e in negative_embeddings))
        cache_key = (query_text.strip().lower(), round(diversity, 2), img_hash,
                     combination_mode, neg_hash, negative_mode)
        pool = self._pool_cache_get(cache_key)

        if pool is None:
            # ── 1. Encode text ────────────────────────────────────────────────────
            if query_text.strip():
                inputs = self.processor(text=[query_text], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_text_features(**inputs)
                    if not isinstance(outputs, torch.Tensor):
                        text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
                    else:
                        text_features = outputs
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                text_np = text_features.cpu().numpy().astype('float32')  # (1, D)
            else:
                text_np = None

            # ── 2. Build list of all normalised query vectors ─────────────────────
            # Each element is a (1, D) float32 array
            all_query_vecs = []
            if text_np is not None:
                all_query_vecs.append(text_np)
            if individual_image_embeddings:          # preferred: per-image embeddings
                all_query_vecs.extend(individual_image_embeddings)
            elif image_embedding is not None:        # legacy: pre-averaged embedding
                all_query_vecs.append(image_embedding)

            if not all_query_vecs:
                return {"results": [], "total": 0, "has_more": False}

            n_els = len(all_query_vecs)

            # ── 3. Compute combined query embedding ───────────────────────────────
            valid_ids   = None   # set by intersection/union paths
            score_by_id = None
            text_embedding = None  # single (1, D) query vector for FAISS + MMR

            if n_els == 1:
                # Only one element — all modes are equivalent
                text_embedding = all_query_vecs[0]

            elif combination_mode == "centroid":
                # Weighted blend: text vs images, then normalise
                if text_np is not None and individual_image_embeddings:
                    avg_img = np.mean(
                        np.concatenate(individual_image_embeddings, axis=0), axis=0, keepdims=True
                    )
                    blended = (1.0 - image_weight) * text_np + image_weight * avg_img
                else:
                    stacked = np.concatenate(all_query_vecs, axis=0)  # (N, D)
                    blended = stacked.mean(axis=0, keepdims=True)
                norm = np.linalg.norm(blended, axis=1, keepdims=True)
                text_embedding = (blended / np.where(norm == 0, 1e-9, norm)).astype('float32')

            elif combination_mode == "pca":
                # First principal component of the query matrix
                stacked = np.concatenate(all_query_vecs, axis=0)  # (N, D)
                _, _, Vt = np.linalg.svd(stacked, full_matrices=False)
                pc1 = Vt[0:1]  # (1, D)
                # Align sign so most query vectors have a positive dot product
                if float((stacked @ pc1.T).mean()) < 0:
                    pc1 = -pc1
                text_embedding = (
                    pc1 / (np.linalg.norm(pc1, axis=1, keepdims=True) + 1e-9)
                ).astype('float32')

            elif combination_mode == "purified":
                # Query mean minus a fraction of the corpus mean.
                # Amplifies dimensions specific to the query and suppresses
                # generic 'art image' dimensions present in the whole dataset.
                stacked = np.concatenate(all_query_vecs, axis=0)
                mean_vec = stacked.mean(axis=0, keepdims=True)
                if self.corpus_mean is not None:
                    mean_vec = mean_vec - 0.4 * self.corpus_mean[np.newaxis, :]
                norm = np.linalg.norm(mean_vec, axis=1, keepdims=True)
                text_embedding = (mean_vec / np.where(norm == 0, 1e-9, norm)).astype('float32')

            elif combination_mode in ("intersection", "union"):
                # Gather candidates from centroid search + per-element searches,
                # then re-score by min (AND) or max (OR) similarity.
                stacked = np.concatenate(all_query_vecs, axis=0)
                mean_vec = stacked.mean(axis=0, keepdims=True)
                norm = np.linalg.norm(mean_vec, axis=1, keepdims=True)
                centroid_vec = (mean_vec / np.where(norm == 0, 1e-9, norm)).astype('float32')

                fetch_k   = min(pool_size * 3, self.index.ntotal)
                per_el_k  = max(100, fetch_k // n_els)
                cand_set  = set()

                # Centroid sweep
                _, idx0 = self.index.search(centroid_vec, fetch_k)
                cand_set.update(
                    int(i) for i in idx0[0] if i >= 0 and int(i) not in self.denylist
                )
                # Per-element sweeps
                for ev in all_query_vecs:
                    _, idx_e = self.index.search(ev, per_el_k)
                    cand_set.update(
                        int(i) for i in idx_e[0] if i >= 0 and int(i) not in self.denylist
                    )

                valid_raw = list(cand_set)

                # Reconstruct + normalise candidate vectors
                cand_vecs = np.vstack(
                    [self.index.reconstruct(int(fid)) for fid in valid_raw]
                ).astype('float32')                          # (M, D)
                norms_c = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
                cand_vecs /= np.where(norms_c == 0, 1e-9, norms_c)

                # Similarity of every candidate to every query element: (M, N_q)
                query_mat = np.concatenate(all_query_vecs, axis=0)  # (N_q, D)
                sim_mat   = cand_vecs @ query_mat.T                  # (M, N_q)

                if combination_mode == "intersection":
                    scores = sim_mat.min(axis=1)   # (M,) — must satisfy ALL
                else:
                    scores = sim_mat.max(axis=1)   # (M,) — satisfies ANY

                order       = np.argsort(-scores)
                valid_ids   = [valid_raw[i] for i in order]
                score_by_id = {valid_raw[i]: float(scores[i]) for i in range(len(valid_raw))}
                text_embedding = centroid_vec    # used for MMR diversity scoring

            else:
                # Unknown mode: fall back to simple mean
                stacked = np.concatenate(all_query_vecs, axis=0)
                mean_vec = stacked.mean(axis=0, keepdims=True)
                norm = np.linalg.norm(mean_vec, axis=1, keepdims=True)
                text_embedding = (mean_vec / np.where(norm == 0, 1e-9, norm)).astype('float32')

            # ── Apply negative embeddings: directed or orthogonal (pre-FAISS) ───
            if negative_embeddings and text_embedding is not None and all_query_vecs:
                pos_mean = np.concatenate(all_query_vecs, axis=0).mean(axis=0).astype('float32')  # (D,)
                if negative_mode == "directed":
                    alpha = 0.35
                    for neg_emb in negative_embeddings:
                        neg_dev = neg_emb[0] - pos_mean
                        text_embedding[0] -= alpha * neg_dev
                    norm = np.linalg.norm(text_embedding[0])
                    text_embedding[0] /= (norm + 1e-9)
                elif negative_mode == "orthogonal":
                    for neg_emb in negative_embeddings:
                        neg_dir = (neg_emb[0] - pos_mean).astype('float32')
                        neg_dir /= (np.linalg.norm(neg_dir) + 1e-9)
                        text_embedding[0] -= (text_embedding[0] @ neg_dir) * neg_dir
                    norm = np.linalg.norm(text_embedding[0])
                    text_embedding[0] /= (norm + 1e-9)

            # ── 4. FAISS search (skipped for intersection/union — already done) ──
            if valid_ids is None:
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

            # ── Penalty mode: penalise candidates similar to negatives (post-FAISS) ──
            if negative_embeddings and negative_mode == "penalty" and valid_ids:
                beta = 0.4
                pen_ids = list(score_by_id.keys())
                if pen_ids:
                    pen_vecs = np.vstack(
                        [self.index.reconstruct(int(fid)) for fid in pen_ids]
                    ).astype('float32')
                    norms_p = np.linalg.norm(pen_vecs, axis=1, keepdims=True)
                    pen_vecs /= np.where(norms_p == 0, 1e-9, norms_p)
                    for neg_emb in negative_embeddings:
                        neg_sims = (pen_vecs @ neg_emb.T).flatten()
                        for i, fid in enumerate(pen_ids):
                            score_by_id[fid] = score_by_id.get(fid, 0.0) - beta * float(neg_sims[i])
                    valid_ids = sorted(valid_ids, key=lambda x: -score_by_id.get(x, 0.0))

            # ── 5. Rank / re-rank to pool_size ───────────────────────────────────
            cluster_sizes   = {}   # faiss_id → Voronoi count
            cluster_members = {}   # faiss_id → [faiss_ids in cluster]

            if diversity > 0.0 and len(valid_ids) > pool_size:
                selected_ids, cluster_sizes, cluster_members = self._mmr_rerank(
                    text_embedding[0], valid_ids, score_by_id, pool_size, diversity
                )
            else:
                selected_ids = valid_ids[:pool_size]
                if len(valid_ids) > pool_size:
                    selected_ids, cluster_sizes, cluster_members = \
                        self._deduplicate_and_voronoi(selected_ids, valid_ids, score_by_id)

            # ── 6. Fetch metadata for the pool ───────────────────────────────────
            meta_by_id = self._lookup_metadata(selected_ids)
            pool = []
            for fid in selected_ids:
                if fid not in meta_by_id:
                    continue
                item = dict(meta_by_id[fid])
                item['score'] = score_by_id.get(fid, 0.0)
                if cluster_sizes:
                    item['cluster_size']       = cluster_sizes.get(fid, 1)
                    members = cluster_members.get(fid, [fid])
                    members_sorted = sorted(
                        members, key=lambda x: score_by_id.get(x, 0.0), reverse=True
                    )
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
