# src/edgar_rag/evals/run_evals.py
"""Retrieval + answer quality metrics against evals/golden.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anthropic

from edgar_rag.agent import ask
from edgar_rag.search import search

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5"

JUDGE = """You grade an answer produced by a retrieval system over bank 10-K filings.

Return ONLY JSON:
{"faithful": true|false, "citations_valid": true|false, "answers_question": true|false, "why": "one sentence"}

faithful: every factual claim is supported by the numbered sources. Any figure, date, or named program not in the sources makes this false.
citations_valid: every [n] marker points to a source that actually supports the adjacent claim.
answers_question: the answer addresses what was asked, rather than adjacent material."""


def load(path="evals/golden.jsonl") -> list[dict]:
    return [json.loads(l) for l in Path(path).open()]


def key(h: dict) -> tuple:
    return (h["ticker"], h["accession"], h["item"], int(h["chunk_index"]))

def eval_retrieval(golden, k=5, qtype=None):
    if qtype is not None:
        golden = [g for g in golden if g.get("question_type") == qtype]
    hits_at_k, rr, misses = 0, 0.0, []
    for g in golden:
        gold = tuple(g["gold_chunk"][f] for f in
                     ("ticker", "accession", "item", "chunk_index"))
        results = search(g["question"], k=k)
        ranks = [i for i, h in enumerate(results, 1) if key(h) == gold]
        if ranks:
            hits_at_k += 1
            rr += 1 / ranks[0]
        else:
            misses.append(g["question"])
    n = len(golden)
    return {f"recall@{k}": hits_at_k / n, "mrr": rr / n, "n": n, "misses": misses[:10]}


def eval_answers(golden: list[dict], limit: int = 20) -> dict:
    scores = {"faithful": 0, "citations_valid": 0, "answers_question": 0}
    n = 0
    for g in golden[:limit]:
        out = ask(g["question"])
        ctx = "\n\n".join(
            f"[{i}] {h['ticker']} {h['filing_date']} {h['item']}\n{h['text'][:1500]}"
            for i, h in enumerate(out["hits"], 1)
        )
        r = client.messages.create(
            model=MODEL, max_tokens=300, system=JUDGE,
            messages=[{"role": "user", "content":
                       f"QUESTION: {g['question']}\n\nSOURCES:\n{ctx}\n\nANSWER:\n{out['answer']}"}],
        )
        txt = "".join(b.text for b in r.content if b.type == "text").strip()
        txt = txt.removeprefix("```json").removesuffix("```").strip()
        try:
            d = json.loads(txt)
        except json.JSONDecodeError:
            continue
        for kk in scores:
            scores[kk] += bool(d.get(kk))
        n += 1
    return {kk: v / n for kk, v in scores.items()} | {"n": n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--out", default="evals/results.json")
    p.add_argument("--type", type=str, default=None)
    a = p.parse_args()

    golden = load()
    res = {"retrieval": eval_retrieval(golden, a.k, a.type)}
    if not a.retrieval_only:
        res["answers"] = eval_answers(golden)

    Path(a.out).write_text(json.dumps(res, indent=2))
    r = res["retrieval"]
    print(f"recall@{a.k}: {r[f'recall@{a.k}']:.2%}   mrr: {r['mrr']:.3f}   n={r['n']}")
    if "answers" in res:
        aa = res["answers"]
        print(f"faithful: {aa['faithful']:.2%}   citations: {aa['citations_valid']:.2%}   "
              f"on-topic: {aa['answers_question']:.2%}   n={aa['n']}")
    if r["misses"]:
        print("\nmisses:")
        for m in r["misses"]:
            print("  " + m[:100])


if __name__ == "__main__":
    main()
