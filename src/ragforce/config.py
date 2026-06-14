"""Typed, layered configuration — the single source of truth, threaded everywhere.

Two layers, env wins over file:
    1. ``config.yaml``  — readable defaults (the source of truth a reviewer edits).
    2. ``.env`` / env vars  — runtime overrides, prefix ``RAG__`` + ``__`` nesting,
       e.g. ``RAG__EMBEDDING__MODEL_NAME=...``.

The nested models below document the full config schema *in code*. They carry
defaults matching ``config.yaml`` so the structure is usable in tests without a
file present.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = "config.yaml"


class EmbeddingCfg(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    device: str = "cpu"
    normalize: bool = True
    max_seq_length: int = 512
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""
    batch_size: int = 64
    # Offline / air-gap support. ``models_dir`` is the local HF cache the models are
    # loaded from; ``local_files_only`` forbids any network fetch (fail instead).
    # Pre-populate the cache once online with ``rag fetch-models``.
    models_dir: str | None = None
    local_files_only: bool = False


class ChunkingCfg(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 50
    min_chunk_size: int = 32


class QdrantCfg(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "forensic_docs"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    upsert_batch_size: int = 128
    recreate_on_ingest: bool = False
    timeout: float = 30.0          # client request timeout (seconds)
    # ANN index tuning (exposed rather than left at library defaults).
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_ef_search: int | None = None   # None -> Qdrant default
    quantization: bool = False          # int8 scalar quantization (cheaper RAM/disk)


class HybridCfg(BaseModel):
    enabled: bool = True
    sparse_model: str = "Qdrant/bm25"
    fusion: str = "rrf"
    top_k: int = 5
    # BM25 average document length — must track chunk_size or length-normalization
    # is miscalibrated. fastembed's default (256) is wrong for 400-token chunks.
    avg_len: int = 400
    # Per-branch candidate depth before fusion: max(top_k * multiplier, min).
    prefetch_multiplier: int = 10
    prefetch_min: int = 50


class RerankCfg(BaseModel):
    """Optional cross-encoder reranker over the fused candidate set (quality lever)."""

    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-base"
    top_n: int = 50    # candidates to rerank before truncating to top_k


class PathsCfg(BaseModel):
    source_dir: str = "data/generated"
    seed_file: str = "data/seeds/seed_snippets.jsonl"
    ground_truth: str = "data/generated/ground_truth.json"


class DatasetCfg(BaseModel):
    num_docs: int = 120
    seed: int = 42
    doc_types: list[str] = Field(
        default_factory=lambda: ["witness_statement", "report", "transcript"]
    )
    formats: list[str] = Field(default_factory=lambda: ["txt", "pdf", "json", "eml"])


class ApiCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    default_top_k: int = 5


class LoggingCfg(BaseModel):
    level: str = "INFO"
    config_file: str = "config/logging.yaml"


class Settings(BaseSettings):
    """Top-level settings object passed to every layer."""

    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    qdrant: QdrantCfg = Field(default_factory=QdrantCfg)
    hybrid: HybridCfg = Field(default_factory=HybridCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    # Path to the YAML config; overridable per-call by load_settings().
    _yaml_path: str = DEFAULT_CONFIG_PATH

    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Precedence (high → low): init args, env, .env, YAML file, field defaults."""
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=cls._yaml_path)
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)


def load_settings(config_path: str | None = None) -> Settings:
    """Build :class:`Settings` from ``config.yaml`` overlaid with env/.env (env wins).

    Resolution order, highest first: explicit kwargs, ``RAG__*`` env vars, ``.env``,
    the YAML file, then field defaults. ``config_path`` (or ``$CONFIG_PATH``) selects
    the YAML file.
    """
    Settings._yaml_path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    return Settings()
