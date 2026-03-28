import os
import json
import sqlite3
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

DATA_DIR = "data/images"
METADATA_FILE = "data/metadata.json"
INDEX_FILE = "data/faiss.index"
MAPPING_FILE = "data/index_mapping.json"
DB_FILE = "data/metadata.db"

MODEL_ID = "openai/clip-vit-base-patch32"

def save_to_sqlite(valid_metadata):
    """Save metadata list to SQLite. Each item must have 'faiss_id'."""
    if not valid_metadata:
        return

    print(f"Saving metadata to SQLite: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)

    # Build schema from all keys across all items
    all_keys = set()
    for item in valid_metadata:
        all_keys.update(item.keys())
    all_keys = sorted(all_keys)

    cols = ", ".join(f'"{k}" TEXT' for k in all_keys if k != "faiss_id")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS images (
            faiss_id INTEGER PRIMARY KEY,
            {cols}
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faiss_id ON images(faiss_id)")

    for item in tqdm(valid_metadata, desc="Writing to SQLite"):
        keys = ["faiss_id"] + [k for k in all_keys if k != "faiss_id"]
        vals = [item.get(k, "") for k in keys]
        placeholders = ",".join("?" * len(keys))
        col_names = ",".join(f'"{k}"' for k in keys)
        conn.execute(f"INSERT OR REPLACE INTO images ({col_names}) VALUES ({placeholders})", vals)

    conn.commit()
    conn.close()
    size_mb = os.path.getsize(DB_FILE) / 1024 / 1024
    print(f"  ✓ SQLite saved ({len(valid_metadata)} rows, {size_mb:.1f} MB)")


def main():
    # 1. Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found: {METADATA_FILE}. Run download_data.py first.")
        return

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    # 2. Load Model
    print("Loading CLIP model...")
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # 3. Generate Embeddings
    print("Generating embeddings...")
    embeddings = []
    valid_metadata = []

    for i, item in enumerate(tqdm(metadata)):
        image_path = os.path.join(DATA_DIR, item['filename'])
        try:
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                img_inputs = inputs.get('pixel_values', inputs)
                outputs = model.get_image_features(pixel_values=img_inputs)

                if not isinstance(outputs, torch.Tensor):
                    if hasattr(outputs, 'pooler_output'):
                        feat = outputs.pooler_output
                        if feat.shape[-1] == model.visual_projection.in_features:
                            image_features = model.visual_projection(feat)
                        else:
                            image_features = feat
                    else:
                        image_features = outputs
                else:
                    image_features = outputs

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())

            # faiss_id = position in the index (= order of insertion)
            item_with_id = dict(item)
            item_with_id['faiss_id'] = len(valid_metadata)
            valid_metadata.append(item_with_id)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if not embeddings:
        print("No embeddings generated.")
        return

    embeddings_np = np.vstack(embeddings).astype('float32')

    # 4. Build FAISS Index
    print(f"Building FAISS index ({embeddings_np.shape[0]} vectors × {embeddings_np.shape[1]} dims)...")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    # 5. Save FAISS Index
    faiss.write_index(index, INDEX_FILE)
    print(f"  ✓ FAISS index saved to {INDEX_FILE} ({os.path.getsize(INDEX_FILE)//1024//1024} MB)")

    # 6. Save metadata to SQLite (primary) + JSON (fallback/compat)
    save_to_sqlite(valid_metadata)

    with open(MAPPING_FILE, "w") as f:
        json.dump(valid_metadata, f, indent=2)
    print(f"  ✓ JSON mapping saved to {MAPPING_FILE}")

    print(f"\nDone! {len(valid_metadata)} images indexed.")


if __name__ == "__main__":
    main()
