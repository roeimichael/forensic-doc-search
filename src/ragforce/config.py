"""Typed, layered configuration — the single source of truth, threaded everywhere.

Two layers, env wins over file:
    1. ``config.yaml``  — readable defaults (the source of truth a reviewer edits).
    2. ``.env`` / env vars  — runtime overrides, prefix ``RAG__`` + ``__`` nesting,
       e.g. ``RAG__EMBEDDING__MODEL_NAME=...``.

The nested models below document the full config schema *in code*. They carry
defaults matching ``config.yaml`` so the structure is usable in tests without a
file present.

STATUS: schema is real; :func:`load_settings` env-over-yaml wiring is a stub for
the implementation step (see TODO).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class HybridCfg(BaseModel):
    enabled: bool = True
    sparse_model: str = "Qdrant/bm25"
    fusion: str = "rrf"
    top_k: int = 5


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
    formats: list[str] = Field(default_factory=lambda: ["txt", "pdf", "json"])


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
    paths: PathsCfg = Field(default_factory=PathsCfg)
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )


def load_settings(config_path: str | None = None) -> Settings:
    """Build :class:`Settings` from ``config.yaml`` overlaid with env/.env (env wins).

    TODO(X2): wire env-over-yaml precedence via ``settings_customise_sources``
    (init > env > dotenv > YAML source > defaults), then assert
    ``embedding.dim`` against the model's reported dimension at startup.
    """
    raise NotImplementedError("config loading is implemented in the next step (X2)")
