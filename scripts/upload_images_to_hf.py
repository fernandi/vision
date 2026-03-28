"""
Upload images from data/images/ to a HuggingFace Hub dataset repo.
Uses upload_large_folder which handles retries, timeouts and resumption.

Images will be accessible at:
  https://huggingface.co/datasets/{repo}/resolve/main/images/{filename}

Usage:
    python scripts/upload_images_to_hf.py --repo FerBar/vision-images
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=str, required=True,
                    help="HuggingFace repo id, e.g. FerBar/vision-images")
parser.add_argument("--images-dir", type=str, default="data/images",
                    help="Local images directory (default: data/images)")
args = parser.parse_args()

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

# Create repo if needed (public dataset)
print(f"Creating/verifying repo: {args.repo}")
create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True, token=token)

images_dir = Path(args.images_dir)
if not images_dir.exists():
    print(f"Images directory not found: {images_dir}")
    exit(1)

n = len(list(images_dir.glob("*.jpg")))
print(f"Found {n} images to upload from {images_dir}")
print("Using upload_large_folder (resumable, timeout-safe)...")

api.upload_large_folder(
    repo_id=args.repo,
    repo_type="dataset",
    folder_path=str(images_dir),
    path_in_repo="images",
)

print(f"\nDone!")
print(f"Public URL: https://huggingface.co/datasets/{args.repo}/resolve/main/images/0.jpg")
