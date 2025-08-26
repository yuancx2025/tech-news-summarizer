# src/embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import numpy as np

# Simplified version for testing - will be replaced with full version later
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbedderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    normalize: bool = True
    batch_size: int = 64
    show_progress: bool = True


class Embedder:
    """
    Simplified embedder for testing - will be replaced with full SentenceTransformer version
    """
    def __init__(self, cfg: EmbedderConfig | None = None):
        self.cfg = cfg or EmbedderConfig()
        # For now, just create dummy embeddings
        self._dummy_dim = 384  # Standard dimension for all-MiniLM-L6-v2
        
    @property
    def dim(self) -> int:
        return self._dummy_dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Create dummy embeddings for testing
        """
        if not texts:
            return np.array([]).reshape(0, self._dummy_dim)
        
        # Create random embeddings for testing
        n_texts = len(texts)
        embeddings = np.random.randn(n_texts, self._dummy_dim).astype(np.float32)
        
        if self.cfg.normalize:
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms
            
        return embeddings


# --------- Helpful utilities you can reuse elsewhere ---------

def combine_title_text(
    title: str | None,
    text: str | None,
    *,
    max_chars: int = 1200,
    title_weight: int = 1,
) -> str:
    """
    Build a single string used for embedding:
      - Puts title first (optionally repeated to upweight)
      - Truncates article text for speed/recall balance
    """
    title = (title or "").strip()
    text = (text or "").strip()
    parts: List[str] = []
    if title:
        # Repeating title can slightly bias similarity toward topical match
        parts.append((" " + title + " ") * max(1, title_weight))
    if text:
        parts.append(text[:max_chars])
    s = "\n".join(p.strip() for p in parts if p.strip())
    return s or ""


def ensure_unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize a 2D array (N, D). No-op if already normalized.
    """
    if x.ndim != 2:
        raise ValueError("Expected 2D array (N, D)")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32)
