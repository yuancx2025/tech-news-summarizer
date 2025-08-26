# model/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple
from time import perf_counter

from rouge_score import rouge_scorer
import numpy as np


@dataclass
class RougeConfig:
    use_stemmer: bool = True
    metrics: Tuple[str, ...] = ("rouge1", "rouge2", "rougeLsum")


def _mk_scorer(cfg: RougeConfig) -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(cfg.metrics, use_stemmer=cfg.use_stemmer)


def rouge_batch(
    predictions: List[str],
    references: List[str],
    cfg: RougeConfig = RougeConfig(),
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean precision/recall/f1 for each ROUGE metric.
    Returns: {metric: {"p": float, "r": float, "f": float}}
    """
    assert len(predictions) == len(references), "pred/refs length mismatch"
    scorer = _mk_scorer(cfg)
    sums = {m: {"p": 0.0, "r": 0.0, "f": 0.0} for m in cfg.metrics}
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # (reference, prediction)
        for m in cfg.metrics:
            sums[m]["p"] += scores[m].precision
            sums[m]["r"] += scores[m].recall
            sums[m]["f"] += scores[m].fmeasure
    return {m: {k: v / max(n, 1) for k, v in sums[m].items()} for m in cfg.metrics}


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
