import os
import json
import argparse
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_NAME = "Mitsua/art-museums-pd-440k"
OUTPUT_DIR = "data/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=10000,
                    help="Number of images to download (default: 10000, use 0 for all)")
args = parser.parse_args()

LIMIT = args.limit if args.limit > 0 else None
limit_display = LIMIT if LIMIT else "all"

print(f"Loading dataset {DATASET_NAME} (streaming)...")
ds = load_dataset(DATASET_NAME, split="train", streaming=True)

print(f"Downloading {limit_display} images to {OUTPUT_DIR}...")
count = 0
metadata = []

for i, item in tqdm(enumerate(ds), total=LIMIT):
    if LIMIT and count >= LIMIT:
        break

    try:
        img = item.get('jpg')
        if img:
            filename = f"{count}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            meta = item.get('json', {})
            meta['id'] = count
            meta['hf_idx'] = i  # real position in HF dataset (for image proxy)
            meta['filename'] = filename
            metadata.append(meta)

            if not os.path.exists(filepath):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(filepath)

            count += 1
    except Exception as e:
        print(f"Error processing item {i}: {e}")

# Save metadata index
with open("data/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Done! Downloaded {count} images.")
