# src/embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required. Install it with:\n"
        "  pip install -U sentence-transformers"
    ) from e


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbedderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    normalize: bool = True
    batch_size: int = 64
    show_progress: bool = True


class Embedder:
    """
    Thin wrapper around SentenceTransformer to:
      - keep config in one place
      - guarantee np.float32 output
      - (optionally) L2-normalize embeddings so dot==cosine downstream
    """

    def __init__(self, cfg: EmbedderConfig | None = None):
        self.cfg = cfg or EmbedderConfig()
        self.model = SentenceTransformer(self.cfg.model_name)

    @property
    def dim(self) -> int:
        # Load a single dummy to query dimension lazily if needed
        v = self.model.encode([""], normalize_embeddings=self.cfg.normalize)
        return int(v.shape[1])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a batch of texts into a (N, D) float32 array.
        """
        arr = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            show_progress_bar=self.cfg.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        # sentence-transformers returns float32 by default when convert_to_numpy=True,
        # but we cast defensively.
        return np.asarray(arr, dtype=np.float32)


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
