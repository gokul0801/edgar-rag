from edgar_rag.ingest import cik_for_ticker, recent_filings, fetch_text

cik = cik_for_ticker("JPM")
f = recent_filings(cik, limit=1)[0]
print(f)
text = fetch_text(cik, f["accession"], f["doc"])
open("/tmp/jpm.txt", "w").write(text)
print(f"{len(text):,} chars written to /tmp/jpm.txt")
