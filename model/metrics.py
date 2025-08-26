# model/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple
from time import perf_counter

import numpy as np

# Note: ROUGE metrics are disabled due to import issues on macOS
# Using fallback overlap metrics instead
ROUGE_AVAILABLE = False


@dataclass
class RougeConfig:
    use_stemmer: bool = True
    metrics: Tuple[str, ...] = ("overlap",)  # Changed to use fallback metric


def _mk_scorer(cfg: RougeConfig):
    """Placeholder function - always returns None since ROUGE is disabled"""
    return None


def rouge_batch(
    predictions: List[str],
    references: List[str],
    cfg: RougeConfig = RougeConfig(),
) -> Dict[str, Dict[str, float]]:
    """
    Compute fallback overlap metrics since ROUGE is not available.
    Returns: {"overlap": {"p": float, "r": float, "f": float}}
    """
    assert len(predictions) == len(references), "pred/refs length mismatch"
    
    print("[metrics] Using fallback overlap metrics (ROUGE disabled)")
    return _fallback_overlap_metrics(predictions, references)


def _fallback_overlap_metrics(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """Simple fallback metrics that provide basic text similarity"""
    def word_overlap(pred_words, ref_words):
        if not ref_words:
            return {"p": 0.0, "r": 0.0, "f": 0.0}
        
        pred_set = set(pred_words.lower().split())
        ref_set = set(ref_words.lower().split())
        
        if not pred_set or not ref_set:
            return {"p": 0.0, "r": 0.0, "f": 0.0}
        
        intersection = pred_set.intersection(ref_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        recall = len(intersection) / len(ref_set) if ref_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"p": precision, "r": recall, "f": f1}
    
    # Compute simple word overlap metrics
    overlap_scores = []
    for pred, ref in zip(predictions, references):
        overlap_scores.append(word_overlap(pred, ref))
    
    # Average the scores
    n = len(overlap_scores)
    avg_scores = {}
    for key in ["p", "r", "f"]:
        avg_scores[key] = sum(score[key] for score in overlap_scores) / max(n, 1)
    
    return {
        "overlap": avg_scores,
        "note": "Fallback metrics (ROUGE disabled due to import issues)"
    }


def lengths(texts: Iterable[str]) -> Dict[str, float]:
    lens = [len(t.split()) for t in texts]
    return {
        "words_avg": float(np.mean(lens)) if lens else 0.0,
        "words_p50": float(np.median(lens)) if lens else 0.0,
        "words_p90": float(np.percentile(lens, 90)) if lens else 0.0,
    }


def compression_ratio(srcs: Iterable[str], preds: Iterable[str]) -> float:
    src_w = sum(len(s.split()) for s in srcs) + 1e-9
    pred_w = sum(len(p.split()) for p in preds)
    return float(pred_w / src_w)


class Timer:
    """Simple latency accumulator."""
    def __init__(self): self.t = []
    def tic(self): self._s = perf_counter()
    def toc(self): self.t.append((perf_counter() - self._s) * 1000.0)  # ms
    def stats(self) -> Dict[str, float]:
        if not self.t: return {"ms_avg": 0.0, "ms_p50": 0.0, "ms_p90": 0.0}
        arr = np.array(self.t, dtype=np.float64)
        return {"ms_avg": float(arr.mean()), "ms_p50": float(np.median(arr)), "ms_p90": float(np.percentile(arr, 90))}
