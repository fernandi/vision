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
        self.metadata_mapping = []
        self.load_error = None  # exposed via /health for debugging


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

        n_indexed = self.index.ntotal
        print(f"Search Engine Ready. ({n_indexed} images indexed)")

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

    def search(self, query_text, k=20):
        if not self.model:
            self.load()

        inputs = self.processor(text=[query_text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            if not isinstance(outputs, torch.Tensor):
                text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            else:
                text_features = outputs

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_embedding = text_features.cpu().numpy().astype('float32')

        distances, indices = self.index.search(text_embedding, k)

        valid_ids = [int(idx) for idx in indices[0] if idx >= 0]
        meta_by_id = self._lookup_metadata(valid_ids)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or int(idx) not in meta_by_id:
                continue
            item = dict(meta_by_id[int(idx)])
            item['score'] = float(distances[0][i])
            results.append(item)

        return results
