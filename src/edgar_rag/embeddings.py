# src/edgar_rag/embeddings.py
"""Bedrock Titan v2 embeddings with batching, retry, and adaptive backoff."""
from __future__ import annotations

import json
import random
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from edgar_rag.config import settings

MODEL_ID = "amazon.titan-embed-text-v2:0"
DIM = 1024
MAX_CHARS = 30_000          # Titan caps ~8k tokens
RETRYABLE = {"ThrottlingException", "TooManyRequestsException",
             "ServiceUnavailableException", "ModelTimeoutException",
             "InternalServerException"}

_rt = boto3.client(
    "bedrock-runtime",
    region_name=settings().bedrock_region,
    config=Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=60,
        max_pool_connections=32,
    ),
)


def _embed_one(text: str, attempts: int = 8) -> list[float]:
    body = json.dumps({
        "inputText": text[:MAX_CHARS],
        "dimensions": DIM,
        "normalize": True,
    })
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = _rt.invoke_model(modelId=MODEL_ID, body=body)
            return json.loads(r["body"].read())["embedding"]
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in RETRYABLE:
                raise
            last = e
            time.sleep(min(2 ** i, 30) + random.uniform(0, 1))
        except Exception as e:  # transient network
            last = e
            time.sleep(min(2 ** i, 30) + random.uniform(0, 1))
    raise RuntimeError(f"embed failed after {attempts} attempts: {last}")


def embed(texts: list[str]) -> list[list[float]]:
    """Serial by design. Use embed_many() for bulk."""
    return [_embed_one(t) for t in texts]


def embed_query(text: str) -> list[float]:
    return _embed_one(text)
