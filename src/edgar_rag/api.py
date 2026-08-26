# src/edgar_rag/api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from edgar_rag.agent import ask
from edgar_rag.search import search

app = FastAPI(title="EDGAR RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    rerank: bool = True


class Source(BaseModel):
    n: int
    ticker: str
    filing_date: str
    item: str
    chunk_index: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(q: str, k: int = 5, ticker: str | None = None,
                    item: str | None = None) -> list[dict]:
    hits = search(q, k=k, ticker=ticker, item=item)
    return [{"ticker": h["ticker"], "filing_date": h["filing_date"],
             "item": h["item"], "chunk_index": h["chunk_index"],
             "text": h["text"][:500]} for h in hits]


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    try:
        out = ask(req.question)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return AskResponse(
        answer=out["answer"],
        sources=[
            Source(n=i, ticker=h["ticker"], filing_date=h["filing_date"],
                   item=h["item"], chunk_index=h["chunk_index"],
                   excerpt=h["text"][:300])
            for i, h in enumerate(out["hits"], 1)
        ],
    )