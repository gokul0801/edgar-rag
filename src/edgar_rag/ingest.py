"""Pull 10-K/10-Q filings from EDGAR and build a section-aware LanceDB index.

EDGAR requires a descriptive User-Agent with real contact info and rate limits
to 10 requests/sec. We stay well under.

The reason this file is longer than a naive loader: 10-Ks have structure, and
chunking across Item boundaries is the single most common thing that makes
filing retrieval bad. Item 1A Risk Factors and Item 7 MD&A are different
documents wearing the same cover.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from edgar import Company, set_identity

# SEC requires a real contact string
set_identity("Gokul Raj gokul@grajconsulting.com")

SECTIONS_OF_INTEREST = {
    "Item 1A": "Risk Factors",
    "Item 3": "Legal Proceedings",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8": "Financial Statements and Supplementary Data",
}

REQUIRED = {"Item 1A", "Item 7"}
MIN_CHARS = {"Item 1A": 15_000, "Item 7": 2_000, "Item 7A": 150, "Item 8": 10_000}

TARGET = 3_000      # chars per chunk (~750 tokens)
OVERLAP = 450       # ~15%
WS = re.compile(r"[\xa0\u2007\u202f\s]+")


@dataclass
class Chunk:
    text: str
    ticker: str
    accession: str
    form_type: str
    filing_date: str
    item: str
    section_name: str
    chunk_index: int


def normalize(t: str) -> str:
    return WS.sub(" ", t).strip()


def get_sections(filing) -> dict[str, str]:
    """Pull items via edgartools; fall back to empty on parse failure."""
    try:
        tenk = filing.obj()
    except Exception:
        return {}
    out = {}
    for key in SECTIONS_OF_INTEREST:
        try:
            body = tenk[key]
        except Exception:
            body = None
        if body:
            out[key] = normalize(str(body))
    return out


def validate(sections: dict[str, str], tag: str) -> None:
    missing = REQUIRED - sections.keys()
    if missing:
        raise ValueError(f"{tag}: missing {sorted(missing)}")
    for item, floor in MIN_CHARS.items():
        if item in sections and len(sections[item]) < floor:
            raise ValueError(f"{tag}: {item} only {len(sections[item]):,} chars")


def chunk_section(body: str) -> list[str]:
    """Sentence-boundary-aware sliding window."""
    if len(body) <= TARGET:
        return [body] if len(body) > 200 else []
    chunks, start = [], 0
    while start < len(body):
        end = min(start + TARGET, len(body))
        if end < len(body):
            cut = body.rfind(". ", start + TARGET // 2, end)
            if cut != -1:
                end = cut + 1
        piece = body[start:end].strip()
        if len(piece) > 200:
            chunks.append(piece)
        if end >= len(body):
            break
        start = end - OVERLAP
    return chunks


def build_chunks(tickers: list[str], per_ticker: int = 3) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    failures: list[str] = []

    for ticker in tickers:
        filings = Company(ticker).get_filings(form="10-K").head(per_ticker)
        for filing in filings:
            tag = f"{ticker} 10-K {filing.filing_date}"
            try:
                sections = get_sections(filing)
                validate(sections, tag)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{tag}: {e}")
                continue

            n = 0
            for item, body in sections.items():
                for j, c in enumerate(chunk_section(body)):
                    all_chunks.append(Chunk(
                        text=c,
                        ticker=ticker,
                        accession=str(filing.accession_no),
                        form_type="10-K",
                        filing_date=str(filing.filing_date),
                        item=item,
                        section_name=SECTIONS_OF_INTEREST[item],
                        chunk_index=j,
                    ))
                    n += 1
            print(f"{tag}: {n} chunks  ({', '.join(f'{k}={len(v):,}' for k, v in sections.items())})")

    if failures:
        print("\nSKIPPED:")
        for x in failures:
            print("  " + x)
    print(f"\n{len(all_chunks)} chunks total")
    return all_chunks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["JPM", "GS", "MS", "C", "BAC", "SCHW"])
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--out", default="data/chunks.jsonl")
    args = p.parse_args()

    chunks = build_chunks(args.tickers, args.years)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
