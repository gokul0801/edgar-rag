# src/edgar_rag/build_index.py
"""chunks.jsonl -> Bedrock embeddings -> LanceDB table. Resumable."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lancedb
import pyarrow as pa

from edgar_rag.config import settings
from edgar_rag.embeddings import DIM, MODEL_ID, _embed_one

SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), DIM)),
    pa.field("text", pa.string()),
    pa.field("ticker", pa.string()),
    pa.field("accession", pa.string()),
    pa.field("form_type", pa.string()),
    pa.field("filing_date", pa.string()),
    pa.field("item", pa.string()),
    pa.field("section_name", pa.string()),
    pa.field("chunk_index", pa.int64()),
    pa.field("embed_model", pa.string()),
])


def embed_all(rows: list[dict], workers: int, cache: Path) -> list[dict]:
    """Embed with bounded concurrency; cache to disk so a crash doesn't cost a rerun."""
    done: dict[str, list[float]] = {}
    if cache.exists():
        with cache.open() as fh:
            for line in fh:
                d = json.loads(line)
                done[d["id"]] = d["v"]
        print(f"resuming: {len(done)} cached")

    todo = [r for r in rows if r["_id"] not in done]
    if todo:
        t0 = time.time()
        with cache.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_embed_one, r["text"]): r for r in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                r = futs[fut]
                v = fut.result()
                done[r["_id"]] = v
                fh.write(json.dumps({"id": r["_id"], "v": v}) + "\n")
                fh.flush()
                if n % 100 == 0 or n == len(todo):
                    rate = n / (time.time() - t0)
                    eta = (len(todo) - n) / rate / 60
                    print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {eta:.1f}m")

    for r in rows:
        r["vector"] = done[r["_id"]]
        r["embed_model"] = MODEL_ID
        del r["_id"]
    return rows

"""
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/chunks.jsonl")
    p.add_argument("--db", default=settings().db_path)
    p.add_argument("--table", default=settings().table_name)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="data/.embed_cache.jsonl")
    a = p.parse_args()

    rows = [json.loads(l) for l in Path(a.inp).open()]
    for r in rows:
        r["_id"] = f"{r['accession']}:{r['item']}:{r['chunk_index']}"
    print(f"{len(rows)} chunks")

    rows = embed_all(rows, a.workers, Path(a.cache))

    db = lancedb.connect(a.db)
    try:
        tbl = db.open_table(a.table)
        tbl.add(data=rows)
        print(f"Successfully appended {len(rows)} chunks to existing table '{a.table}'.")
    except Exception:
        # Fallback: If running on a fresh path or bucket, create it fresh
        print(f"Table '{a.table}' not found or empty. Building table from scratch...")
        tbl = db.create_table(a.table, data=rows, schema=SCHEMA, mode="overwrite")
    #tbl = db.create_table(a.table, data=rows, schema=SCHEMA, mode="overwrite")
    tbl.create_fts_index("text", replace=True)
    print(f"{tbl.count_rows()} rows -> {a.db}/{a.table}")
    print(f"cache retained at {a.cache} (delete to force re-embed)")
"""

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/chunks.jsonl")
    p.add_argument("--db", default=settings().db_path)
    p.add_argument("--table", default=settings().table_name)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="data/.embed_cache.jsonl")
    a = p.parse_args()

    # 1. Load ALL original text rows
    rows = [json.loads(l) for l in Path(a.inp).open()]
    
    # 2. Re-map the IDs exactly how they were cached
    for r in rows:
        r["_id"] = f"{r['accession']}:{r['item']}:{r['chunk_index']}"
    print(f"Total rows to compile: {len(rows)}")

    # 3. Read the cache file directly into memory
    cache_path = Path(a.cache)
    cached_vectors = {}
    if cache_path.exists():
        with cache_path.open() as fh:
            for line in fh:
                d = json.loads(line)
                cached_vectors[d["id"]] = d["v"]
        print(f"Loaded {len(cached_vectors)} vectors from disk cache.")

    # 4. Bind the vectors directly to your rows list
    final_rows = []
    for r in rows:
        if r["_id"] in cached_vectors:
            r["vector"] = cached_vectors[r["_id"]]
            r["embed_model"] = MODEL_ID
            del r["_id"] # clean up temporary ID string
            final_rows.append(r)
        else:
            print(f"Warning: Chunk {r['_id']} missing from cache!")

    print(f"Compiling {len(final_rows)} complete rows with vectors...")

    # 5. Overwrite the table using the verified data array
    db = lancedb.connect(a.db)
    tbl = db.create_table(a.table, data=final_rows, schema=SCHEMA, mode="overwrite")
    tbl.create_fts_index("text", replace=True)
    
    print(f"🎉 Success! {tbl.count_rows()} rows -> {a.db}/{a.table}")


if __name__ == "__main__":
    main()
