# src/edgar_rag/evals/build_golden.py
"""Bootstrap a golden set: sample chunks -> LLM writes a question each chunk uniquely answers."""
from __future__ import annotations

import json
import random
from pathlib import Path

import anthropic
import lancedb

from edgar_rag.config import settings

client = anthropic.Anthropic(api_key=settings().anthropic_api_key)
MODEL = "claude-sonnet-4-5"

SYS = """You write evaluation questions for a financial-filings retrieval system.

Given one chunk from a bank's 10-K, write ONE question that this chunk uniquely answers.

Rules:
- The question must be answerable from this chunk alone.
- It must be specific enough that a generic filings search wouldn't surface a dozen equally-good chunks. Prefer concrete mechanisms, named programs, specific metrics.
- Do NOT quote the chunk. Use natural analyst phrasing.
- Include the company name only if the answer is company-specific.
- Avoid questions answerable from general knowledge.

Return ONLY JSON: {"question": "...", "answer_gist": "...", "quality": "high"|"low"}
Set quality "low" if the chunk is boilerplate, a fragment, or too generic to anchor a question."""


def sample_chunks(n: int = 60) -> list[dict]:
    tbl = lancedb.connect(settings().db_path).open_table(settings().table_name)
    rows = tbl.to_pandas().to_dict("records")
    # stratify across ticker + item so the set isn't all JPM risk factors
    buckets: dict[tuple, list] = {}
    for r in rows:
        buckets.setdefault((r["ticker"], r["item"]), []).append(r)
    out, keys = [], list(buckets)
    random.seed(7)
    while len(out) < n and keys:
        for k in list(keys):
            b = buckets[k]
            if not b:
                keys.remove(k)
                continue
            out.append(b.pop(random.randrange(len(b))))
            if len(out) >= n:
                break
    return out


def gen_question(chunk: dict) -> dict | None:
    r = client.messages.create(
        model=MODEL, max_tokens=400, system=SYS,
        messages=[{"role": "user", "content": chunk["text"][:4000]}],
    )
    txt = "".join(b.text for b in r.content if b.type == "text").strip()
    txt = txt.removeprefix("```json").removesuffix("```").strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        return None
    if d.get("quality") != "high":
        return None
    return {
        "question": d["question"],
        "answer_gist": d["answer_gist"],
        "gold_chunk": {
            "ticker": chunk["ticker"],
            "accession": chunk["accession"],
            "item": chunk["item"],
            "chunk_index": int(chunk["chunk_index"]),
        },
        "gold_text": chunk["text"],
    }


def main() -> None:
    out_path = Path("evals/golden.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    kept = []
    for i, c in enumerate(sample_chunks(60), 1):
        q = gen_question(c)
        if q:
            kept.append(q)
            print(f"{i:>3} ✓ {q['question'][:80]}")
        else:
            print(f"{i:>3} ✗ skipped (low quality)")
    with out_path.open("w") as fh:
        for q in kept:
            fh.write(json.dumps(q) + "\n")
    print(f"\n{len(kept)} questions -> {out_path}")


if __name__ == "__main__":
    main()