"""
Mirror Glane's images from the public Mitsua/art-museums-pd-440k dataset.

Museum servers block (Art Institute of Chicago) or have lost ~1 image in 5, so
the site serves its own copies. Two WebP renditions per indexed work, keyed by
its faiss_id:

    t/{faiss_id}.webp   grid thumbnail, 480 px wide
    h/{faiss_id}.webp   full-page view, dataset resolution (~843 px short side)

Shards are streamed straight from Hugging Face (nothing stored but the output),
encoded on every core, and written to a folder or an S3-compatible bucket.

    python scripts/mirror_images.py --out dir:D:/glane-mirror --shards 41
    python scripts/mirror_images.py --out s3://glane-images --shards 1-41

S3 / Cloudflare R2 credentials: S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY (pip install boto3). The job is resumable: finished
shards are recorded in --state, re-run the same command to continue.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED

import warnings

from PIL import Image, ImageOps

warnings.filterwarnings("ignore", message="Corrupt EXIF data")   # harmless, frequent in museum files

DATASET = "https://huggingface.co/datasets/Mitsua/art-museums-pd-440k/resolve/main/ArtMuseumsPD_{:04d}.tar"
MAPPING_URL = "https://huggingface.co/datasets/{repo}/resolve/main/index_mapping.json"
THUMB_WIDTH, THUMB_QUALITY = 480, 72
HD_MAX_EDGE, HD_QUALITY = 1600, 80
UA = {"User-Agent": "glane-mirror/1.0"}
Image.MAX_IMAGE_PIXELS = 200_000_000


# ── ImageID → faiss ids, streamed out of the 442 MB index_mapping.json ───────
def iter_json_array(path, chunk=1 << 24):
    decoder = json.JSONDecoder()
    buf, pos, started = "", 0, False
    with open(path, encoding="utf-8") as f:
        while True:
            data = f.read(chunk)
            buf, pos = buf[pos:] + data, 0
            if not started:
                pos, started = buf.index("[") + 1, True
            while True:
                while pos < len(buf) and buf[pos] in " \r\n\t,":
                    pos += 1
                if pos >= len(buf) or buf[pos] == "]":
                    break
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break                                   # object cut by the chunk: read more
                yield obj
                pos = end
            if not data:
                return


def load_id_map(state, mapping_path, repo):
    cache = os.path.join(state, "imageid_to_faiss.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    if not mapping_path:
        mapping_path = os.path.join(state, "index_mapping.json")
        if not os.path.exists(mapping_path):
            print(f"downloading index_mapping.json from {repo}…", flush=True)
            urllib.request.urlretrieve(MAPPING_URL.format(repo=repo), mapping_path)
    id_map = {}
    for row in iter_json_array(mapping_path):
        id_map.setdefault(str(row["ImageID"]), []).append(int(row.get("faiss_id", row.get("id"))))
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
    total = sum(len(v) for v in id_map.values())
    print(f"id map: {total} indexed works, {len(id_map)} distinct ImageIDs", flush=True)
    return id_map


# ── Encoding (runs in worker processes) ─────────────────────────────────────
def _webp(im, quality):
    out = io.BytesIO()
    im.save(out, "WEBP", quality=quality, method=4)
    return out.getvalue()


def render(data):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(im.size) > HD_MAX_EDGE:
        im.thumbnail((HD_MAX_EDGE, HD_MAX_EDGE), Image.LANCZOS)
    hd = _webp(im, HD_QUALITY)
    if im.width > THUMB_WIDTH:
        im = im.resize((THUMB_WIDTH, round(im.height * THUMB_WIDTH / im.width)), Image.LANCZOS)
    return _webp(im, THUMB_QUALITY), hd


# ── Outputs ──────────────────────────────────────────────────────────────────
class DirStore:
    def __init__(self, root):
        self.root = root
        self.bytes = 0

    def put(self, key, data):
        self.bytes += len(data)
        path = os.path.join(self.root, *key.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)


class S3Store:
    def __init__(self, bucket, prefix):
        import boto3
        from botocore.config import Config
        self.client = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"), region_name="auto",
                                   config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "adaptive"}))
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.bytes = 0

    def put(self, key, data):
        self.bytes += len(data)
        self.client.put_object(
            Bucket=self.bucket, Key=f"{self.prefix}/{key}" if self.prefix else key, Body=data,
            ContentType="image/webp", CacheControl="public, max-age=31536000, immutable")


def open_store(spec):
    if spec.startswith("dir:"):
        return DirStore(spec[4:])
    if spec.startswith("s3://"):
        bucket, _, prefix = spec[5:].partition("/")
        return S3Store(bucket, prefix)
    raise SystemExit("--out must be dir:PATH or s3://bucket[/prefix]")


# ── One shard ────────────────────────────────────────────────────────────────
def mirror_shard(n, id_map, store, encoders, uploaders, in_flight):
    stats = {"images": 0, "unindexed": 0, "failed": 0, "written": []}
    uploads = []

    def store_result(ids, fut):
        try:
            thumb, hd = fut.result()
        except Exception:
            stats["failed"] += 1
            return
        for fid in ids:
            uploads.append(uploaders.submit(store.put, f"t/{fid}.webp", thumb))
            uploads.append(uploaders.submit(store.put, f"h/{fid}.webp", hd))
            stats["written"].append(fid)

    pending = {}
    with urllib.request.urlopen(urllib.request.Request(DATASET.format(n), headers=UA), timeout=120) as resp:
        archive = tarfile.open(fileobj=resp, mode="r|")
        for member in archive:
            base, _, ext = member.name.rpartition(".")
            if ext.lower() not in ("jpg", "jpeg", "png", "webp") or not member.isfile():
                continue
            stats["images"] += 1
            ids = id_map.get(os.path.basename(base))
            if not ids:
                stats["unindexed"] += 1
                continue
            data = archive.extractfile(member).read()
            pending[encoders.submit(render, data)] = ids
            if len(pending) >= in_flight:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    store_result(pending.pop(fut), fut)
            while len(uploads) > in_flight * 4:
                uploads.pop(0).result()
    for fut in list(pending):
        store_result(pending.pop(fut), fut)
    for up in uploads:
        up.result()
    return stats


def parse_shards(spec):
    shards = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        shards.extend(range(int(a), int(b or a) + 1))
    return shards


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="dir:PATH or s3://bucket[/prefix]")
    p.add_argument("--shards", default="1-41", help="e.g. 1-41, 41, 3,7")
    p.add_argument("--state", default="mirror_state", help="progress, id map and report folder")
    p.add_argument("--mapping", help="local index_mapping.json (downloaded otherwise)")
    p.add_argument("--index-repo", default=os.environ.get("HF_INDEX_REPO", "FerBar/vision-index"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()

    os.makedirs(args.state, exist_ok=True)
    done_path = os.path.join(args.state, "done.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    id_map = load_id_map(args.state, args.mapping, args.index_repo)
    store = open_store(args.out)
    started = time.time()

    with ProcessPoolExecutor(args.workers) as encoders, ThreadPoolExecutor(16) as uploaders:
        for n in parse_shards(args.shards):
            if n in done:
                print(f"shard {n:02d}: already done", flush=True)
                continue
            for attempt in range(1, 4):
                t0 = time.time()
                try:
                    stats = mirror_shard(n, id_map, store, encoders, uploaders, in_flight=args.workers * 3)
                    break
                except Exception as e:
                    print(f"shard {n:02d}: attempt {attempt} failed ({e})", flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(10 * attempt)
            with open(os.path.join(args.state, f"written_{n:02d}.txt"), "w") as f:
                f.write("\n".join(map(str, stats["written"])))
            done.add(n)
            with open(done_path, "w") as f:
                json.dump(sorted(done), f)
            print(f"shard {n:02d}: {len(stats['written'])} written, {stats['unindexed']} not in the index, "
                  f"{stats['failed']} unreadable, {time.time() - t0:.0f}s", flush=True)

    # Resource use of this run (encoder processes included), to check hosting costs.
    t = os.times()
    cpu = t.user + t.system + t.children_user + t.children_system
    print(f"usage: {time.time() - started:.0f}s wall, {cpu:.0f} CPU-seconds, "
          f"{store.bytes / 1e9:.3f} GB written", flush=True)

    written = set()
    for name in os.listdir(args.state):
        if name.startswith("written_"):
            written.update(int(x) for x in open(os.path.join(args.state, name)).read().split())
    indexed = {fid for ids in id_map.values() for fid in ids}
    missing = sorted(indexed - written)
    with open(os.path.join(args.state, "missing.txt"), "w") as f:
        f.write("\n".join(map(str, missing)))
    print(f"total: {len(written)} / {len(indexed)} indexed works mirrored; "
          f"{len(missing)} still missing (list in {args.state}/missing.txt)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
