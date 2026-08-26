from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Bedrock
    bedrock_region: str = "us-east-2"
    model_synthesis: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    model_classify: str = "anthropic.claude-haiku-4-5-20251001-v1:0"

    # API keys
    anthropic_api_key: str = ""

    embed_model: str = "BAAI/bge-small-en-v1.5"
    # Storage — swap to s3://<bucket>/lance once deployed
    db_path: str = "data/lancedb"
    table_name: str = "filings"

    # Retrieval
    top_k: int = 8
    max_agent_loops: int = 2

    # Ops
    request_timeout_s: float = 55.0
    log_level: str = "INFO"


@lru_cache
def settings() -> Settings:
    return Settings()
