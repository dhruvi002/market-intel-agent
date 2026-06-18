"""Central settings loaded from environment / .env file."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ─────────────────────────────────────────────────────────
    environment: str = Field("development", description="development | staging | production")
    log_level: str = Field("INFO")

    # ── Postgres ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://mia:mia_dev@localhost:5432/market_intel"
    )

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://:redis_dev@localhost:6379/0")

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = Field("http://localhost:6333")

    # ── MinIO ────────────────────────────────────────────────────────────────
    minio_endpoint: str = Field("localhost:9000")
    minio_access_key: str = Field("mia_minio")
    minio_secret_key: SecretStr = Field("minio_dev_secret")
    minio_secure: bool = Field(False)
    minio_bucket_filings: str = Field("sec-filings")

    # ── LLM providers ────────────────────────────────────────────────────────
    gemini_api_key: SecretStr = Field(...)
    groq_api_key: SecretStr = Field(...)
    cerebras_api_key: SecretStr | None = Field(None)
    llm_provider: str | None = Field(
        None,
        description=(
            "Pin all get_llm() calls to one provider (gemini | groq | cerebras). "
            "None → full Gemini→Groq→Cerebras fallback chain."
        ),
    )

    # ── Search ───────────────────────────────────────────────────────────────
    tavily_api_key: SecretStr = Field(...)
    tavily_cache_ttl_days: int = Field(7)

    # ── Langfuse ─────────────────────────────────────────────────────────────
    langfuse_host: str = Field("http://localhost:3000")
    langfuse_public_key: str | None = Field(None)
    langfuse_secret_key: SecretStr | None = Field(None)

    # ── Agent behaviour ──────────────────────────────────────────────────────
    max_iterations: int = Field(3, description="Critic self-correction iteration cap")
    max_evidence_chunks: int = Field(20, description="Max chunks passed to Summarizer")

    # ── Retrieval ────────────────────────────────────────────────────────────
    embedding_model: str = Field("BAAI/bge-large-en-v1.5")
    reranker_model: str = Field("BAAI/bge-reranker-v2-m3")
    nli_model: str = Field("cross-encoder/nli-deberta-v3-base")
    nli_entailment_threshold: float = Field(
        0.5, description="Min entailment probability for Citation.is_verified=True"
    )
    qdrant_collection: str = Field("filings")
    bm25_top_k: int = Field(50)
    dense_top_k: int = Field(50)
    rerank_top_k: int = Field(10)

    # ── SQL Generator ────────────────────────────────────────────────────────
    sql_max_rows: int = Field(50, description="Max rows returned by sql_generator_node")

    # ── EDGAR ────────────────────────────────────────────────────────────────
    edgar_user_agent: str = Field(
        "MarketIntelAgent/0.1 worksofdhruvi@gmail.com",
        description="Required by EDGAR fair-access policy",
    )
    edgar_request_delay_s: float = Field(0.11, description="≥0.1 s between EDGAR requests")

    # ── Evaluation (Phase 8) ─────────────────────────────────────────────────
    eval_golden_path: str | None = Field(
        None, description="Override path to golden_set.jsonl (defaults to packaged set)"
    )
    eval_top_k: int = Field(10, description="Retrieval cut-off for Recall/Precision/nDCG@k")
    eval_bootstrap_samples: int = Field(
        10_000, description="Resamples for the 95% CI bootstrap in mia_eval.stats"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
