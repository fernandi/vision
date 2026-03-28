"""
Upload FAISS index + index_mapping.json to a HuggingFace Hub dataset repo.

Usage:
    python scripts/upload_index_to_hf.py --repo fernandi/vision-index

The repo will be created as a dataset if it doesn't exist.
Files are uploaded via the HF Hub API (handles Git LFS automatically).
"""
import argparse
import os
from huggingface_hub import HfApi, create_repo

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=str, required=True,
                    help="HuggingFace repo id, e.g. fernandi/vision-index")
parser.add_argument("--private", action="store_true",
                    help="Make the repo private (needs HF_TOKEN with write access)")
args = parser.parse_args()

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

# Create the dataset repo if needed
print(f"Creating/verifying repo: {args.repo}")
create_repo(args.repo, repo_type="dataset", private=args.private,
            exist_ok=True, token=token)

files_to_upload = [
    ("data/faiss.index", "faiss.index"),
    ("data/index_mapping.json", "index_mapping.json"),
]

for local_path, remote_path in files_to_upload:
    if not os.path.exists(local_path):
        print(f"WARNING: {local_path} not found, skipping.")
        continue
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {local_path} ({size_mb:.1f} MB) → {args.repo}/{remote_path} ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=args.repo,
        repo_type="dataset",
        token=token,
    )
    print(f"  ✓ Done")

print("\nAll files uploaded successfully.")
print(f"Index URL: https://huggingface.co/datasets/{args.repo}")
