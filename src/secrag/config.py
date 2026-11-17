"""Runtime settings, read from environment variables (and .env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Settings:
    sec_user_agent: str = field(default_factory=lambda: _env("SEC_USER_AGENT", ""))
    data_dir: Path = field(default_factory=lambda: Path(_env("SECRAG_DATA_DIR", "data")))

    # LLM (any OpenAI-compatible server: vLLM, Ollama, TGI, a hosted API ...)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "EMPTY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    # Retrieval
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_query_prefix: str = field(
        default_factory=lambda: _env(
            "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
        )
    )
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-base"))
    chunk_words: int = field(default_factory=lambda: _env_int("CHUNK_WORDS", 300))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    n_candidates: int = field(default_factory=lambda: _env_int("N_CANDIDATES", 50))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    bm25_weight: float = field(default_factory=lambda: float(_env("BM25_WEIGHT", "1.0")))

    # Agents
    max_evidence: int = field(default_factory=lambda: _env_int("MAX_EVIDENCE", 16))
    max_verify_rounds: int = field(default_factory=lambda: _env_int("MAX_VERIFY_ROUNDS", 2))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def xbrl_dir(self) -> Path:
        return self.data_dir / "xbrl"
