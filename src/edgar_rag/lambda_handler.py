# src/edgar_rag/lambda_handler.py
"""Thin Lambda entrypoint. Cold-start caches the table connection."""
from __future__ import annotations

import json
import os

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from edgar_rag.agent import ask
        _agent = ask
    return _agent


def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        question = (body.get("question") or "").strip()
        if not 3 <= len(question) <= 500:
            return _resp(400, {"error": "question must be 3-500 chars"})

        out = _get_agent()(question)
        return _resp(200, {
            "answer": out["answer"],
            "sources": [
                {"n": i, "ticker": h["ticker"], "filing_date": h["filing_date"],
                 "item": h["item"], "chunk_index": h["chunk_index"],
                 "excerpt": h["text"][:300]}
                for i, h in enumerate(out["hits"], 1)
            ],
        })
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}")
        return _resp(500, {"error": "internal error"})


def _resp(code: int, body: dict) -> dict:
    return {
        "statusCode": code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }
