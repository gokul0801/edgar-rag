# src/edgar_rag/agent.py
from __future__ import annotations

import json
from typing import Annotated, Literal, TypedDict

import anthropic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from edgar_rag.config import settings
from edgar_rag.search import search

client = anthropic.Anthropic(api_key=settings().anthropic_api_key)
MODEL = "claude-sonnet-4-5"

class Plan(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    item: str | None = None
    rewritten: str


class State(TypedDict):
    question: str
    plan: Plan | None
    hits: list[dict]
    answer: str


def _json_call(system: str, user: str) -> dict:
    r = client.messages.create(
        model=MODEL, max_tokens=512, system=system,
        messages=[{"role": "user", "content": user}],
    )
    txt = "".join(b.text for b in r.content if b.type == "text")
    return json.loads(txt.strip().removeprefix("```json").removesuffix("```").strip())


def plan(state: State) -> State:
    sys = (
        "Extract search parameters from a question about bank 10-K filings. "
        "Return ONLY JSON: {\"tickers\": [], \"item\": null, \"rewritten\": \"\"}. "
        "tickers: any of JPM GS MS C BAC SCHW mentioned, else []. "
        "item: one of 'Item 1A','Item 3','Item 7','Item 7A','Item 8' if the question "
        "clearly targets risk factors, litigation, MD&A, market risk, or financials; banks discuss market risk inside MD&A, so prefer Item 7 over Item 7A ;else null. "
        "rewritten: the question as a dense retrieval query, no company names."
    )
    return {**state, "plan": Plan(**_json_call(sys, state["question"]))}


def retrieve(state: State) -> State:
    p = state["plan"]
    hits: list[dict] = []
    for t in (p.tickers or [None]):
        hits += search(p.rewritten, k=6, ticker=t, item=p.item)
    return {**state, "hits": hits}


def synthesize(state: State) -> State:
    ctx = "\n\n".join(
        f"[{i}] {h['ticker']} {h['filing_date']} {h['item']}\n{h['text']}"
        for i, h in enumerate(state["hits"], 1)
    )
    sys = (
        "Answer using ONLY the numbered sources. Cite inline as [1], [2]. "
        "If the sources don't support an answer, say so. Be concise. "
        "Never state a figure that isn't in the sources."
    )
    r = client.messages.create(
        model=MODEL, max_tokens=1024, system=sys,
        messages=[{"role": "user",
                   "content": f"{ctx}\n\nQuestion: {state['question']}"}],
    )
    return {**state, "answer": "".join(b.text for b in r.content if b.type == "text")}


g = StateGraph(State)
g.add_node("plan", plan)
g.add_node("retrieve", retrieve)
g.add_node("synthesize", synthesize)
g.set_entry_point("plan")
g.add_edge("plan", "retrieve")
g.add_edge("retrieve", "synthesize")
g.add_edge("synthesize", END)
app = g.compile()


def ask(q: str) -> dict:
    return app.invoke({"question": q, "plan": None, "hits": [], "answer": ""})


if __name__ == "__main__":
    import sys
    out = ask(" ".join(sys.argv[1:]))
    print(out["answer"])
    print("\n--- sources ---")
    for i, h in enumerate(out["hits"], 1):
        print(f"[{i}] {h['ticker']} {h['filing_date']} {h['item']} #{h['chunk_index']}")
