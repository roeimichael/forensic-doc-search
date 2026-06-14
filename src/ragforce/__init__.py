"""ragforce — on-prem RAG pipeline for forensic document search.

Layers (ingestion → serving):
    loaders → chunking → embedding → store → pipeline   (ingestion path)
    api / eval / ui                                      (serving + measurement)

Everything is config-driven via :mod:`ragforce.config`; all model inference runs
locally and the vector store is self-hosted (Qdrant). See docs/02_architecture.md.
"""

__version__ = "0.1.0"
