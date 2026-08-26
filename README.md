# EDGAR RAG

Retrieval-augmented Q&A over SEC 10-K filings for the 6 largest US banks, with inline citations back to the source filing.

Filings are pulled from EDGAR, split by 10-K item, validated, and chunked. **Amazon Bedrock (Titan Text v2)** embeds them into a **LanceDB** index that lives in **S3** — an embedded library, not a hosted vector database, so the index never leaves the account. At query time a **LangGraph** agent plans the search, retrieves with hybrid vector + BM25 fusion and cross-encoder reranking, and synthesizes a cited answer. It runs locally under **FastAPI** and deploys to AWS as an **arm64 Docker container on Lambda** behind an IAM-authed Function URL, provisioned with **Terraform**.

Built as a production-shaped system rather than a demo: every stage has a gate, retrieval quality is measured against a golden set, and a failing eval breaks the build.

```
question → plan → embed → hybrid retrieve → rerank → synthesize → cited answer
```

## Why it's built this way

Most RAG demos wire an LLM to a vector store and stop. The parts that actually determine whether the system works are upstream and downstream of the model:

- **Ingest can fail silently.** An early version of the section parser dropped two of six banks and emitted table-of-contents rows as content — 49 chunks where there should have been thousands. Nothing crashed. The fix was a validation gate that halts ingest before anything reaches the embedding model.
- **Vector search alone is not enough.** Exact-term queries (`CECL`, `Level 3`, `90+ days past due`) fail against dense retrieval and succeed against BM25. Hybrid + reranking is what closes the gap.
- **Without evals, "it works" is an opinion.** Retrieval is measured against a golden set of question/chunk pairs, split by question type.

## Architecture

| Stage | Component | Notes |
|---|---|---|
| Ingest | `edgartools` → JSONL | Section extraction, validation gate, chunking |
| Embed | Bedrock Titan Text v2 (1024-dim) | Same model for documents and queries |
| Store | LanceDB on S3 | Embedded library, no server; index lives in your own bucket |
| Retrieve | Vector + BM25 → RRF → cross-encoder rerank | Reranker is local-only; disabled in Lambda |
| Orchestrate | LangGraph: `plan → retrieve → synthesize` | Planner sets metadata filters from intent |
| Serve | FastAPI locally, container Lambda in AWS | arm64, Function URL with IAM auth |
| Infra | Terraform | IAM, Lambda, Function URL |

### Data flow at query time

1. `plan()` — an LLM call turns the raw question into structured search params: which tickers, which 10-K item, and a rewritten dense-retrieval query with company names stripped (they're redundant once you're filtering on `ticker`).
2. `embed_query()` — Bedrock Titan returns a 1024-dim vector. Same model that embedded the filings, so query and document vectors share a space.
3. `search()` — LanceDB reads index pages and data fragments from S3 over HTTP. Vector search and BM25 run in process, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder.
4. `synthesize()` — retrieved chunks are numbered and passed to the LLM, which must cite inline and may not assert a figure absent from the sources.

The LLM never touches S3. Retrieval happens entirely between the two model calls, which means you control exactly which chunks the model can see.

### Why RRF

Cosine distance (0–2, lower better) and BM25 (unbounded, higher better) live on incomparable scales, and normalizing them requires constants that don't transfer from one document set to another. RRF ignores the scores and fuses on rank position instead — scale-free, no calibration. A document ranked highly by *both* retrievers beats one ranked first by a single retriever, which is exactly the behaviour you want from hybrid search.

### Fusion vs. reranking

Two distinct stages, often conflated:

- **RRF** merges two ranked lists using rank position only. Never reads the text. Free.
- **Reranking** feeds each `(query, chunk)` pair through a cross-encoder that reads both together, catching whether a passage actually answers *this* question rather than just sitting in the right topic. Can't be precomputed — fine on 100 candidates, impossible on 6,586.

## Retrieval quality

Measured against a golden set of 49 question/chunk pairs, generated from sampled chunks and manually reviewed.

| Slice | recall@5 | MRR | n |
|---|---|---|---|
| All | 71.43% | 0.512 | 49 |
| Narrative | 67.74% | 0.506 | 31 |
| Numeric | 77.78% | 0.520 | 18 |

Numeric questions outperform narrative ones, which inverts the usual expectation. The reason is that numeric questions carry distinctive tokens (`Level 3`, `Retail Services`, `90+ days past due`) that BM25 matches exactly, while narrative questions ("what factors contribute to information security risk") are phrased generically enough to match a dozen chunks equally well. Narrative is the weaker side and it's a ranking problem, not a retrieval one.

Caveats worth stating: the golden set is synthetic, so questions inherit vocabulary from the chunks they were generated against and recall is optimistic. Some questions blend both modes ("what was the deposit rate and how did it affect X") and are forced into one bucket. At n=18, a single chunk moves the numeric figure by 5.6 points.

`pytest` enforces a floor on recall and MRR so a retrieval regression fails the build rather than shipping quietly.

## Repo layout

```
src/edgar_rag/
  config.py          pydantic-settings; env-driven
  ingest.py          EDGAR fetch → sections → validate → chunk → JSONL
  embeddings.py      Bedrock Titan client, retry + adaptive backoff
  build_index.py     JSONL → embeddings → LanceDB (resumable, disk-cached)
  search.py          hybrid retrieval: vector + BM25 → RRF → rerank
  agent.py           LangGraph: plan → retrieve → synthesize
  api.py             FastAPI
  lambda_handler.py  Lambda entrypoint
  evals/             golden set generation + metrics
infra/main.tf        IAM, Lambda, Function URL
tests/               eval gate
```

## Running it

```bash
# ingest and index
uv run python -m edgar_rag.ingest --tickers JPM GS MS C BAC SCHW --years 3
uv run python -m edgar_rag.build_index --workers 4

# query
uv run python -m edgar_rag.search "interest rate risk exposure" --ticker JPM
uv run python -m edgar_rag.agent "How does JPM manage interest rate risk?"

# evals
uv run python -m edgar_rag.evals.run_evals --retrieval-only

# serve locally
uv run uvicorn edgar_rag.api:app --reload
```

### Deploy

```bash
docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
  --output type=docker -t edgar-rag .
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/edgar-rag:latest

cd infra && terraform apply
```

Two packaging notes that cost real time:

- The zip deploy path is not viable — `pyarrow` and LanceDB's Rust binary put the package at ~390MB against Lambda's 250MB unzipped limit. Container images allow 10GB.
- `buildx` attaches provenance attestations by default, which wraps the push in an OCI image index. Lambda rejects it. Hence `--provenance=false --sbom=false`.

## Design decisions

**Bedrock Titan over local sentence-transformers.** Removes torch from the Lambda runtime entirely — package drops from ~800MB to ~390MB and cold starts stop paying to load model weights. Measured against the golden set before committing: recall was identical (71.43%), MRR moved 2 points, which is noise at n=49.

**Embedded LanceDB on S3 over a hosted vector DB.** The index never leaves your own account. For a regulated institution that's the whole argument — no Pinecone, no third-party data residency question. LanceDB is a library, not a server: it makes HTTP range requests to S3 and does the math in process.

**Ingest writes JSONL; a separate step embeds it.** The gate is structural rather than a raised exception. A bad parse can't reach the encoder because the encoder is a different process reading a different file.

**Model and ingest version are columns in the table.** Changing the embedding model means a new table and a config alias flip, never a mixed-vector table. Makes eval reproducible and rollback trivial.

## Known limitations

- **Precise numeric lookups from tables.** Dense retrieval struggles when the answer is a figure in a table and the question doesn't contain that figure. The right fix is routing numeric questions to XBRL structured data rather than text chunks.
- **Reranking is disabled in Lambda.** The cross-encoder is a separate ~400MB model. Bedrock Rerank is the path to restoring it in the deployed path.
- **The Anthropic API is a third-party egress.** Bedrock and S3 stay inside AWS; synthesis currently does not. Switching to Bedrock Claude closes it and makes the system single-cloud.
- **Filings are chunked at 3,000 chars.** Large for dense retrieval. Parent-window expansion (return neighbours alongside each hit) is the cheapest untested lever on narrative recall.
