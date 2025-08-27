# src/recommender.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try FAISS, fall back to brute-force search if unavailable
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False


# -------- Utils --------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(x: str | pd.Timestamp) -> datetime:
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().astimezone(timezone.utc)
    if isinstance(x, datetime):
        return x.astimezone(timezone.utc)
    # Try pandas first, then fallback
    try:
        dt = pd.to_datetime(x, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return datetime.fromisoformat(x).astimezone(timezone.utc)


def _days_old(published_at: datetime, now: Optional[datetime] = None) -> float:
    now = now or _now_utc()
    return max(0.0, (now - published_at).total_seconds() / 86400.0)


def _safe_load_list(val) -> List[str]:
    """Accepts list, JSON string list, comma-sep string, or None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    s = str(val).strip()
    if not s:
        return []
    # JSON array?
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            arr = json.loads(s)
            if isinstance(arr, dict):  # e.g., {"keywords": ["a","b"]}
                arr = list(arr.values())[0]
            return [str(v).strip() for v in arr if str(v).strip()]
        except Exception:
            pass
    # fallback comma split
    return [t.strip() for t in s.split(",") if t.strip()]


def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


# -------- Config --------

@dataclass
class RecommenderConfig:
    metadata_path: str = "data/processed/articles_clean.csv"  # or parquet/jsonl
    embeddings_path: str = "data/embeddings/embeddings.npy"   # L2-normalized preferred
    index_path: str = "data/faiss/news.index"                  # optional; falls back if missing
    id_col: str = "id"
    text_cols: Tuple[str, ...] = ("title",)
    publisher_col: str = "publisher"
    published_at_col: str = "published_at"
    topic_col: Optional[str] = "topic"                         # can be None
    entities_col: Optional[str] = "entities"
    keywords_col: Optional[str] = "keywords"

    # Retrieval / ranking knobs
    k_candidates: int = 100
    n_return: int = 10
    tau_recency_days: float = 10.0
    lambda_diversity: float = 0.25

    # Weights
    w_sim: float = 1.0
    w_rec: float = 0.2
    # redundancy is implemented via MMR; w_red acts as additional penalty if desired
    w_red: float = 0.0

    # Staleness filter (optional)
    max_age_days: Optional[int] = None  # e.g., 90 for “fresh only”


# -------- Index wrapper --------

class _FaissOrNumpyIndex:
    def __init__(self, vecs: np.ndarray, use_faiss: bool, faiss_index: Optional[object] = None):
        self.vecs = vecs.astype(np.float32, copy=False)
        self.use_faiss = use_faiss and _HAS_FAISS and faiss_index is not None
        self.faiss_index = faiss_index
        if not self.use_faiss:
            # Pre-normalize for cosine/IP search
            self.vecs = _normalize_rows(self.vecs)

    @classmethod
    def from_paths(cls, emb_path: str, index_path: Optional[str] = None) -> "_FaissOrNumpyIndex":
        vecs = np.load(emb_path, mmap_mode="r")
        vecs = vecs.astype(np.float32)
        faiss_index = None
        use_faiss = False
        if _HAS_FAISS and index_path and os.path.exists(index_path):
            try:
                faiss_index = faiss.read_index(index_path)  # type: ignore
                use_faiss = True
            except Exception:
                faiss_index = None
                use_faiss = False
        return cls(vecs, use_faiss, faiss_index)

    def search(self, q_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q_vecs = q_vecs.astype(np.float32)
        if self.use_faiss:
            # assume vectors are already normalized for IP index
            scores, idxs = self.faiss_index.search(q_vecs, k)  # type: ignore
            return scores, idxs
        # Numpy brute-force cosine similarity
        qn = _normalize_rows(q_vecs)
        sims = qn @ self.vecs.T  # (Q, N)
        idxs = np.argpartition(-sims, kth=min(k, sims.shape[1]-1), axis=1)[:, :k]
        # Gather and sort exact top-k
        top_scores = np.take_along_axis(sims, idxs, axis=1)
        order = np.argsort(-top_scores, axis=1)
        idxs = np.take_along_axis(idxs, order, axis=1)
        scores = np.take_along_axis(sims, idxs, axis=1)
        return scores, idxs


# -------- Recommender --------

class Recommender:
    """
    Lightweight content-based recommender:
    - Candidate gen via FAISS (or NumPy fallback) on article embeddings
    - Re-rank with recency boost + MMR diversity
    - Simple explanations from entity/keyword overlap and freshness
    """

    def __init__(self, cfg: RecommenderConfig):
        self.cfg = cfg
        self.meta = self._load_metadata(cfg.metadata_path)
        self.id_to_idx: Dict[str, int] = self._build_id_index(self.meta[cfg.id_col].tolist())
        self.emb = np.load(cfg.embeddings_path, mmap_mode="r").astype(np.float32)
        # Safety: match rows
        if self.emb.shape[0] != len(self.meta):
            raise ValueError(
                f"Embeddings rows ({self.emb.shape[0]}) != metadata rows ({len(self.meta)}). "
                "Ensure they are aligned and built from the same snapshot."
            )
        # Build or load index
        self.index = _FaissOrNumpyIndex.from_paths(cfg.embeddings_path, cfg.index_path)

        # Column handles
        self._col_id = cfg.id_col
        self._col_pub = cfg.publisher_col
        self._col_time = cfg.published_at_col
        self._col_topic = cfg.topic_col
        self._col_entities = cfg.entities_col
        self._col_keywords = cfg.keywords_col

    # ---- Loading ----
    def _load_metadata(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".jsonl") or path.endswith(".json"):
            df = pd.read_json(path, lines=path.endswith(".jsonl"))
        else:
            df = pd.read_csv(path)
        # normalize types
        if self._col_time in df.columns:
            df[self._col_time] = pd.to_datetime(df[self._col_time], utc=True, errors="coerce")
        return df

    def _build_id_index(self, ids: Iterable) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i, _id in enumerate(ids):
            out[str(_id)] = i
        return out

    # ---- Public API ----
    def get_related_by_id(
        self,
        article_id: str,
        n: Optional[int] = None,
        k_candidates: Optional[int] = None,
        fresh_only: bool = False,
    ) -> List[Dict]:
        """Return top-n related articles with explanations."""
        n = n or self.cfg.n_return
        k_candidates = k_candidates or self.cfg.k_candidates

        if str(article_id) not in self.id_to_idx:
            raise KeyError(f"Article id {article_id} not found")

        q_idx = self.id_to_idx[str(article_id)]
        q_vec = self.emb[q_idx : q_idx + 1]

        # Candidate gen
        scores, idxs = self.index.search(q_vec, k_candidates + 1)  # include self
        cand_idxs = idxs[0].tolist()
        cand_sims = scores[0].tolist()

        # Remove self
        if q_idx in cand_idxs:
            j = cand_idxs.index(q_idx)
            cand_idxs.pop(j)
            cand_sims.pop(j)

        # Filter staleness if asked
        if fresh_only and self.cfg.max_age_days is not None:
            now = _now_utc()
            keep = []
            keep_scores = []
            for i, s in zip(cand_idxs, cand_sims):
                dt = self._get_time(i)
                if pd.isna(dt):
                    continue
                if _days_old(_parse_dt(dt), now) <= self.cfg.max_age_days:
                    keep.append(i)
                    keep_scores.append(s)
            cand_idxs, cand_sims = keep, keep_scores

        # Re-rank: base score (sim + recency), then MMR for diversity
        base_scores = []
        now = _now_utc()
        for i, sim in zip(cand_idxs, cand_sims):
            dt = self._get_time(i)
            days = _days_old(_parse_dt(dt), now) if not pd.isna(dt) else 365.0
            rec_boost = math.exp(-days / self.cfg.tau_recency_days) if self.cfg.tau_recency_days > 0 else 1.0
            score = self.cfg.w_sim * float(sim) + self.cfg.w_rec * float(rec_boost)
            base_scores.append(score)

        # Select N with MMR-style diversity
        selected = self._mmr_select(
            query_idx=q_idx,
            cand_idxs=cand_idxs,
            base_scores=np.array(base_scores, dtype=np.float32),
            n=n,
            lambda_div=self.cfg.lambda_diversity,
        )

        # Build results + explanations
        out = []
        for i in selected:
            meta_row = self.meta.iloc[i]
            sim = cand_sims[cand_idxs.index(i)] if i in cand_idxs else float("nan")
            dt = self._get_time(i)
            days = _days_old(_parse_dt(dt), now) if not pd.isna(dt) else None
            ex = self._explain_pair(q_idx, i, sim, days)
            out.append(
                {
                    "id": str(meta_row[self._col_id]),
                    "title": meta_row.get("title", ""),
                    "url": meta_row.get("url", ""),
                    "publisher": meta_row.get(self._col_pub, ""),
                    "published_at": str(meta_row.get(self._col_time, "")),
                    "similarity": float(sim) if sim == sim else None,
                    "score": float(base_scores[cand_idxs.index(i)]) if i in cand_idxs else None,
                    "recency_days": float(days) if days is not None else None,
                    "explanation": ex,
                }
            )
        return out

    # ---- Internals ----
    def _get_time(self, idx: int):
        return self.meta.iloc[idx].get(self._col_time, None)

    def _pair_sim(self, i: int, j: int) -> float:
        vi = self.emb[i].astype(np.float32)
        vj = self.emb[j].astype(np.float32)
        vi = vi / (np.linalg.norm(vi) + 1e-12)
        vj = vj / (np.linalg.norm(vj) + 1e-12)
        return float(np.dot(vi, vj))

    def _mmr_select(
        self,
        query_idx: int,
        cand_idxs: List[int],
        base_scores: np.ndarray,
        n: int,
        lambda_div: float,
    ) -> List[int]:
        """Maximal Marginal Relevance (MMR) with recency already in base_scores."""
        if not cand_idxs:
            return []
        cand_order = np.argsort(-base_scores)  # start from best base score
        ordered = [cand_idxs[i] for i in cand_order.tolist()]
        ordered_scores = base_scores[cand_order]

        selected: List[int] = []
        selected_set = set()

        while len(selected) < min(n, len(ordered)):
            best_idx = None
            best_mmr = -1e9
            for i, (cand, base) in enumerate(zip(ordered, ordered_scores)):
                if cand in selected_set:
                    continue
                if not selected:
                    mmr = float(base)
                else:
                    # penalize similarity to already selected
                    max_sim = max(self._pair_sim(cand, s) for s in selected)
                    mmr = float(base) - lambda_div * float(max_sim)
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = cand
            if best_idx is None:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
        return selected

    def _explain_pair(self, q_idx: int, c_idx: int, sim: Optional[float], days: Optional[float]) -> str:
        q_row = self.meta.iloc[q_idx]
        c_row = self.meta.iloc[c_idx]

        q_ents = set(_safe_load_list(q_row.get(self._col_entities))) if self._col_entities in self.meta.columns else set()
        c_ents = set(_safe_load_list(c_row.get(self._col_entities))) if self._col_entities in self.meta.columns else set()
        q_keys = set(_safe_load_list(q_row.get(self._col_keywords))) if self._col_keywords in self.meta.columns else set()
        c_keys = set(_safe_load_list(c_row.get(self._col_keywords))) if self._col_keywords in self.meta.columns else set()

        overlap = [t for t in (q_ents & c_ents) or (q_keys & c_keys)]
        overlap = list(overlap)[:3]  # cap
        parts = []
        if overlap:
            parts.append(f"Shared: {', '.join(overlap)}")
        if sim is not None and sim == sim:
            parts.append(f"similarity {sim:.2f}")
        if days is not None:
            if days < 1:
                parts.append("published <1 day ago")
            else:
                parts.append(f"{int(round(days))} days old")
        if not parts:
            parts.append("Topically similar and fresh")
        return " · ".join(parts)


# -------- Convenience factory --------

def load_default_recommender(**overrides) -> Recommender:
    cfg = RecommenderConfig(**overrides)
    return Recommender(cfg)
