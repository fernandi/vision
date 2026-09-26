"""
Build metadata.db (SQLite) from index_mapping.json.

The search server otherwise loads the whole 442 MB JSON into memory (~0.9 GB of
RAM) and, without SQLite, cannot run its exclusion list (stamp sheets, ceramic
shards). With metadata.db it reads only the rows a search needs.

    python scripts/build_metadata_db.py --mapping mirror_state/index_mapping.json --out metadata.db

Same schema as scripts/index_data.py (faiss_id INTEGER PRIMARY KEY, other
fields TEXT), plus two INTEGER bitmasks, period and technique, read from the
descriptions by app/backend/facets.py for the search filters. Publish the file
and point METADATA_DB_URL at it.

To add or refresh the filter columns of an existing file without the JSON:

    python scripts/build_metadata_db.py --facets-only metadata.db
"""
import argparse
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mirror_images import iter_json_array   # noqa: E402
from app.backend.facets import extract      # noqa: E402

COLUMNS = ["Author", "ImageID", "ImageURL", "License", "Title", "URL", "captionEn", "captionJp",
           "filename", "id", "source"]


def add_facets(con):
    """(Re)compute the period and technique columns of every row."""
    have = {row[1] for row in con.execute("PRAGMA table_info(images)")}
    for col in ("period", "technique"):
        if col not in have:
            con.execute(f"ALTER TABLE images ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
    rows = con.execute("SELECT faiss_id, Title, captionEn FROM images").fetchall()
    con.executemany("UPDATE images SET period = ?, technique = ? WHERE faiss_id = ?",
                    [(*extract(title, caption), fid) for fid, title, caption in rows])
    con.commit()
    counted = con.execute("SELECT SUM(period > 0), SUM(technique > 0), COUNT(*) FROM images").fetchone()
    print(f"facets: {counted[0]} rows with a period, {counted[1]} with a technique, of {counted[2]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping")
    p.add_argument("--out", default="metadata.db")
    p.add_argument("--facets-only", metavar="DB", help="only add the filter columns to an existing file")
    args = p.parse_args()

    if args.facets_only:
        t0 = time.time()
        con = sqlite3.connect(args.facets_only)
        add_facets(con)
        con.execute("VACUUM")
        con.close()
        print(f"done in {time.time() - t0:.0f}s")
        return
    if not args.mapping:
        p.error("--mapping is required")

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
    add_facets(con)
    con.execute("VACUUM")
    con.close()
    os.replace(tmp, args.out)
    print(f"{n} rows → {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
