# 10-Day Plan: Ship a Production-Grade Agentic RAG Service

**Aug 14 – Aug 23, 2026.** 4+ hrs/day, ~45 hours total. Ends the day before the road trip.

---

## What you're building

A deployed, publicly demonstrable agentic RAG service over SEC filings, with the full production wrapper: containerized, IaC-provisioned, CI/CD with eval gates, distributed tracing, and per-request cost accounting.

**Why this shape.** You already have LangGraph, RAG, Pydantic, and agentic patterns. What you can't point to in an interview is a deployed system with operational rigor around it — and that's precisely what the Capco GenAI Architect posting means by "MLOps pipelines for deployment, monitoring, and lifecycle management," and what the Google FDE posting means by "LLM-native metrics (tokens/sec, cost-per-request) and granular tracing."

**Domain: SEC EDGAR filings.** Public data, genuinely unstructured, and financial — reinforces your positioning. Critically: nothing from Citi. Not the data, not the code, not the architecture diagrams. Build this on personal hardware, personal accounts, outside work hours.

**Cloud: AWS.** You already know it, so the ten days go into the ops layer rather than fighting an unfamiliar console. Bedrock is the piece that's new and worth having — it's the enterprise LLM story every regulated client asks about, and it's currently missing from your resume.

---

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI, uvicorn, streaming responses |
| Orchestration | LangGraph |
| Structured output | Pydantic + `instructor` |
| Vector store | LanceDB on S3 — serverless, no idle cost |
| Models | Amazon Bedrock (Claude) + Anthropic API direct — deliberately multi-provider |
| Tracing | OpenTelemetry → Langfuse (cloud free tier) |
| Metrics | CloudWatch EMF + custom metrics |
| Secrets | SSM Parameter Store (free tier; Secrets Manager bills per secret) |
| IaC | Terraform |
| CI/CD | GitHub Actions → OIDC role assumption (no long-lived keys) |
| Runtime | Lambda container image + Function URL with response streaming |

---

## Day 1 (Fri) — Skeleton and ingestion

Repo, project structure, dependency management with `uv`. Pull 20–30 10-K and 10-Q filings from EDGAR's API. Write the chunking strategy — section-aware, not naive fixed-size; 10-Ks have structure (Item 1A Risk Factors, Item 7 MD&A) and throwing it away is the single most common RAG mistake.

Embed and load into Qdrant. Get a bare retrieval query working from a script.

**Done when:** you can query "what did Company X say about supply chain risk" from the CLI and get relevant chunks back with source citations.

---

## Day 2 (Sat) — Agent graph

LangGraph state machine with three or four nodes: query classification → retrieval (with query rewriting) → synthesis → self-check. Pydantic models on every boundary, `instructor` for structured extraction.

Add one genuinely agentic behavior — a retrieval node that decides it needs a second search and loops. That's the difference between a RAG pipeline and an agentic system, and it's what the JD language is asking about.

**Done when:** the graph runs end to end, state is inspectable, and the loop demonstrably fires on a multi-hop question.

---

## Day 3 (Sun) — Evals

This is the day most people skip and the day that makes the project credible.

Build a golden dataset — 40–50 question/answer pairs over your filings, hand-labeled. Write three evaluators: retrieval hit rate (is the right chunk in the top-k), faithfulness (is the answer grounded in retrieved text), and structured output validity (does it parse).

Run it as a script, get a baseline number, write it down.

**Done when:** `make eval` prints a scorecard and you know your current numbers.

---

## Day 4 (Mon) — FastAPI service

Wrap the graph in FastAPI. Streaming endpoint via SSE. Request/response models, health check, readiness probe. Proper error handling — timeouts, provider failures, malformed output.

Add resilience: retries with backoff, a circuit breaker on the model provider, and a request timeout that actually cancels work. Bedrock throttles aggressively on burst — handle `ThrottlingException` properly rather than letting it surface as a 500.

**Done when:** `curl` against localhost streams a grounded answer, and killing your network mid-request fails cleanly rather than hanging.

---

## Day 5 (Tue) — Observability

OpenTelemetry instrumentation across every graph node — span per node, with attributes for tokens in/out, model, latency, retrieval scores. Export to Langfuse.

Then the metrics layer: CloudWatch Embedded Metric Format for requests, tokens, latency percentiles, and **cost per request** computed from token counts and current Bedrock pricing. EMF lets you emit structured logs that CloudWatch turns into metrics automatically — no separate metrics endpoint to run. This cost-per-request piece is the specific thing the Google FDE posting named, and almost nobody builds it.

**Done when:** you can open a trace, see the full agent path with per-node timings, and read cost-per-request off a metrics endpoint.

---

## Day 6 (Wed) — Containerize and Terraform

Multi-stage Dockerfile, non-root user, slim final image. Build locally, verify.

Terraform: ECR repository, Lambda function from container image, Function URL with streaming invoke mode, IAM role with least-privilege Bedrock and S3 access, SSM parameters, CloudWatch log group, S3 bucket for the LanceDB index. Everything in code — no console clicking.

Keep the image slim. Lambda container cold starts scale with image size and import time — lazy-import LangGraph and the embedding model inside the handler rather than at module level. Measure the cold start and write the number down; being able to say "cold start was 11s, got it to 3.4s by deferring imports and pre-warming the LanceDB connection" is a better interview answer than never having hit the problem.

**Done when:** `terraform apply` from scratch produces a working deployed service, and `terraform destroy` cleans it up completely.

---

## Day 7 (Thu) — CI/CD with eval gates

GitHub Actions: lint → unit tests → build image → **run evals against the built image** → push to ECR → update Lambda function code.

Authenticate with OIDC role assumption rather than storing AWS keys in GitHub secrets. It's the correct pattern, it takes twenty minutes, and it's the kind of detail a security-conscious client notices.

The eval gate is the interesting part. Set thresholds; if faithfulness drops below baseline, the pipeline fails and the deploy doesn't happen. That's LLM-specific CD, and it's the concrete answer to "lifecycle management" in the JD.

**Done when:** a push to main deploys automatically, and a deliberately bad prompt change gets blocked by the gate.

---

## Day 8 (Fri) — Cost and performance

Semantic caching on the retrieval layer. Measure the hit rate.

Load test with `locust` or `k6` — find where latency degrades, tune Lambda memory (which also scales CPU) and concurrency. Record p50/p95/p99, warm and cold separately.

Model routing: Claude Haiku for classification, Sonnet for synthesis. Measure the cost delta and write it down — "cut cost per request 60% by routing classification to Haiku" is an interview sentence.

**Done when:** you have before/after numbers on cost and latency.

---

## Day 9 (Sat) — Monitoring and failure modes

CloudWatch dashboard: request rate, error rate, p95 latency, token spend, eval scores over time. Alarms on error rate and on cost spike via a Budgets action.

Then deliberately break things and document what happens: Bedrock throttling, malformed model output, empty retrieval, expired credentials. Fix what fails badly.

**Done when:** the dashboard is live and you have a short runbook of failure modes and responses.

---

## Day 10 (Sun) — Write it up

README with an architecture diagram, the design decisions and why, the eval numbers, the cost and latency figures. A short demo video or a live URL.

Then a technical write-up — 1,200 words on one hard problem you solved. Eval gates in CD, or cost-aware model routing, or section-aware chunking for financial documents. Post it. The Capco posting explicitly lists thought leadership as a bonus, and it's the cheapest credential on that list.

**Done when:** someone who's never met you can read the repo and understand what you built and why.

---

## What this gives you in an interview

| Requirement you couldn't previously evidence | What you can now say |
|---|---|
| "Production-grade AI from conception to launch" | Deployed, live URL, CI/CD |
| "MLOps pipelines for deployment, monitoring, lifecycle" | Terraform, Actions, eval gates, dashboards |
| "LLM-native metrics, tokens/sec, cost-per-request" | Built it, has numbers |
| "Architecting AI systems on cloud platforms" | Lambda, Bedrock, S3, Terraform, OIDC |
| "Granular tracing and state management" | OTel spans per graph node |
| "Thought leadership" | Published write-up |

---

## Practical notes

**Cost.** Everything here scales to zero. Lambda free tier covers 1M requests; S3 storage for a few hundred MB is cents; SSM Parameter Store standard tier is free; CloudWatch at this volume is free; Langfuse and LanceDB cost nothing. The only real spend is Bedrock tokens during development and eval runs — realistically **$10–25 for the ten days**.

Set an AWS Budget alarm at $25 on Day 1 anyway. Eval runs hit the API in loops and it's easy to leave one running.

**The architectural decision worth explaining.** Choosing S3-backed LanceDB over managed Postgres with pgvector wasn't just cost — the corpus is read-heavy and rarely updated, so paying for an always-on database buys nothing. That reasoning is exactly the kind of thing an architect interview is probing for.

**Scope risk.** Days 1–5 are the critical path. If you slip, cut Day 8 (cost/perf tuning) before you cut Day 3 (evals) or Day 7 (CI gates) — those two are what make it look like production rather than a portfolio project.

**Citi separation.** Personal machine, personal GitHub, personal cloud account, public data only. Given the outside-activity considerations already in play, keep the boundary clean and obvious.
