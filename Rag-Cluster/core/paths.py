"""Repository-relative defaults for the optional RAG examples."""

from __future__ import annotations

import os
from pathlib import Path


RAG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = RAG_ROOT / "docs"
DEFAULT_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_LOCAL_MODEL = os.environ.get("RAG_LOCAL_MODEL", "")
DEFAULT_INDEX_PATH = DEFAULT_DOCS_DIR / "FAISS-INDEX"
DEFAULT_OPTIONS_JSON = RAG_ROOT / "json_files" / "suzuki" / "dry_sum_suzuki.json"


def configured_path(name: str, default: Path) -> str:
    """Read a path override from the environment without embedding local paths."""

    return os.environ.get(name, str(default))
