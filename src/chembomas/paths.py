"""Repository-relative paths used by public entry points."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root(scope: str) -> Path:
    """Return ``data/wet`` or ``data/dry`` for a named experiment scope."""
    if scope not in {"wet", "dry"}:
        raise ValueError("scope must be 'wet' or 'dry'")
    return REPO_ROOT / "data" / scope


def results_root(scope: str) -> Path:
    if scope not in {"wet", "dry"}:
        raise ValueError("scope must be 'wet' or 'dry'")
    return REPO_ROOT / "results" / scope
