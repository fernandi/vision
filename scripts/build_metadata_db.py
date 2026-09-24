"""
Build metadata.db (SQLite) from index_mapping.json.

The search server otherwise loads the whole 442 MB JSON into memory (~0.9 GB of
RAM) and, without SQLite, cannot run its exclusion list (stamp sheets, ceramic
shards). With metadata.db it reads only the rows a search needs.

    python scripts/build_metadata_db.py --mapping mirror_state/index_mapping.json --out metadata.db

Same schema as scripts/index_data.py (faiss_id INTEGER PRIMARY KEY, other
fields TEXT). Publish the file and point METADATA_DB_URL at it.
"""
import argparse
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirror_images import iter_json_array   # noqa: E402

COLUMNS = ["Author", "ImageID", "ImageURL", "License", "Title", "URL", "captionEn", "captionJp",
           "filename", "id", "source"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", default="metadata.db")
    args = p.parse_args()

    t0 = time.time()
    tmp = args.out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA journal_mode = OFF")
    con.execute("PRAGMA synchronous = OFF")
    cols = ", ".join(f'"{c}" TEXT' for c in COLUMNS)
    con.execute(f"CREATE TABLE images (faiss_id INTEGER PRIMARY KEY, {cols})")
    insert = f"INSERT INTO images (faiss_id, {', '.join(f'"{c}"' for c in COLUMNS)}) VALUES ({', '.join('?' * (len(COLUMNS) + 1))})"

    batch, n = [], 0
    for row in iter_json_array(args.mapping):
        batch.append([int(row["faiss_id"])] + [None if row.get(c) is None else str(row[c]) for c in COLUMNS])
        if len(batch) == 5000:
            con.executemany(insert, batch)
            n += len(batch)
            batch.clear()
    con.executemany(insert, batch)
    n += len(batch)
    con.commit()
    con.execute("VACUUM")
    con.close()
    os.replace(tmp, args.out)
    print(f"{n} rows → {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
