# scripts/migrate_to_s3.py
"""Copy local LanceDB table to S3, verify, then report."""
from __future__ import annotations

import argparse
import subprocess

import lancedb

from edgar_rag.config import settings
from edgar_rag.embeddings import embed_query


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="lancedb")
    p.add_argument("--local", default="data/lancedb")
    p.add_argument("--table", default=settings().table_name)
    a = p.parse_args()

    uri = f"s3://{a.bucket}/{a.prefix}"
    print(f"syncing {a.local} -> {uri}")
    subprocess.run(
        ["aws", "s3", "sync", a.local, uri, "--delete"],
        check=True,
    )

    print("verifying...")
    db = lancedb.connect(uri, storage_options={"region": settings().bedrock_region})
    tbl = db.open_table(a.table)
    print(f"  rows: {tbl.count_rows()}")

    qv = embed_query("interest rate risk management")
    hits = tbl.search(qv).limit(3).to_list()
    for h in hits:
        print(f"  [{h['ticker']} {h['filing_date']} {h['item']}] {h['_distance']:.3f}")

    print("\nFTS index does not survive sync reliably — rebuilding")
    tbl.create_fts_index("text", replace=True)
    print("done")


if __name__ == "__main__":
    main()
