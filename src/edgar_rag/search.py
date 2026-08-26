# src/edgar_rag/search.py
import argparse
import lancedb
import os
from edgar_rag.embeddings import embed_query
from edgar_rag.config import settings

MODEL = settings().embed_model

_ce = None
def _reranker():
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder   # local import
        _ce = CrossEncoder("BAAI/bge-reranker-base")
    return _ce

VEC_WEIGHT = 0.5 # Give semantic vector search 40% influence
FTS_WEIGHT = 0.5 # Give keyword full-text search 60% influence (crucial for SEC)
K = 60


def search(q, k=5, ticker=None, item=None, rerank=os.getenv("ENABLE_RERANK", "true").lower() == "true"):
    tbl = lancedb.connect(settings().db_path).open_table(settings().table_name)
    #qv = SentenceTransformer(settings().embed_model).encode(q, normalize_embeddings=True)
    qv = embed_query(q)
    where = " AND ".join(f"{f} = '{v}'" for f, v in
                         (("ticker", ticker), ("item", item)) if v)

    vec = tbl.search(qv).limit(k * 4)
    fts = tbl.search(q, query_type="fts").limit(k * 4)
    if where:
        vec, fts = vec.where(where, prefilter=True), fts.where(where, prefilter=True)

    # reciprocal rank fusion
    scores = {}
    for rank, r in enumerate(vec.to_list()):
        score_increment = VEC_WEIGHT * (1 / (K + rank))
        scores.setdefault(r["text"], [r, 0])[1] += score_increment
    for rank, r in enumerate(fts.to_list()):
        score_increment = FTS_WEIGHT * (1 / (K + rank))
        scores.setdefault(r["text"], [r, 0])[1] += score_increment

    ranked = sorted(scores.values(), key=lambda x: -x[1])
    cands = [r for r, _ in ranked[:100]]

    if not rerank:
        return cands[:k]

    ce = _reranker()
    ce_scores = ce.predict([(q, r["text"]) for r in cands])
    for r, s in zip(cands, ce_scores):
        r["_rerank_score"] = float(s)
    return sorted(cands, key=lambda r: -r["_rerank_score"])[:k]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--ticker")
    p.add_argument("--item")
    p.add_argument("-k", type=int, default=5)
    a = p.parse_args()
    for r in search(a.query, k=a.k, ticker=a.ticker, item=a.item):
        score = r.get("_rerank_score", r.get("_distance", 0))
        print(f"[{r['ticker']} {r['filing_date']} {r['item']} #{r['chunk_index']}] "
              f"score={score:.3f}")
        print("  " + r["text"][:220].replace("\n", " ") + "…\n")
