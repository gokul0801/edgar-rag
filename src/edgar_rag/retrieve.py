"""Retrieval over the LanceDB index. Day 1 deliverable: prove this works."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import lancedb
from sentence_transformers import SentenceTransformer

from edgar_rag.config import settings


@dataclass
class Hit:
    text: str
    citation: str
    ticker: str
    section_name: str
    score: float


@lru_cache
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings().embed_model)


@lru_cache
def _table():
    return lancedb.connect(settings().lance_uri).open_table("filings")


def search(query: str, k: int | None = None, ticker: str | None = None) -> list[Hit]:
    cfg = settings()
    vec = _embedder().encode(query, normalize_embeddings=True).tolist()

    q = _table().search(vec).limit(k or cfg.top_k)
    if ticker:
        q = q.where(f"ticker = '{ticker}'")

    return [
        Hit(
            text=r["text"],
            citation=r["citation"],
            ticker=r["ticker"],
            section_name=r["section_name"],
            score=1.0 - r["_distance"],
        )
        for r in q.to_list()
    ]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "what are the main interest rate risks disclosed"
    for i, h in enumerate(search(q), 1):
        print(f"\n[{i}] {h.citation}  ({h.section_name})  score={h.score:.3f}")
        print(h.text[:320].replace("\n", " ") + "...")
