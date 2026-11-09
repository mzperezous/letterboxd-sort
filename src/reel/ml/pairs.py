"""Double features, v0: a transparent heuristic pair scorer.

A good double bill is *related but not redundant*: films that talk to each other (shared
themes), ideally from different angles (different director, era, country), at a watchable
combined length. So we score candidate pairs with interpretable features:

    theme     cosine of keyword-only TF-IDF vectors (what the films are about)
    overall   cosine of the full content embedding
    contrast  bonus for different director / decade / country (fresh angle)
    redundant penalty for same collection (sequels) or same director + same decade
    runtime   penalty when the combined runtime is long
    quality   shrunk vote average, so both halves are worth watching

Every feature is returned alongside the score, so results are explainable ("shared: heist,
double cross; contrast: different decade"). These same features become the inputs to a learned
pair model (logistic regression / gradient boosting) once there are labeled double bills.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reel.lake import Lake
from reel.ml.content import ContentModel, build_documents

WEIGHTS = {
    "theme": 1.0,
    "overall": 0.5,
    "contrast": 0.15,
    "redundant": -0.6,
    "runtime": -0.2,
    "quality": 0.3,
}


@dataclass
class PairCandidate:
    movie_id: int
    title: str
    score: float
    features: dict[str, float]
    shared_keywords: list[str]


class DoubleFeatureScorer:
    def __init__(self, lake: Lake, model: ContentModel):
        self.model = model
        self.docs = build_documents(lake)
        rows = lake.query(
            f"""SELECT movie_id, title, release_year, runtime_min, vote_average, vote_count,
                       collection_id FROM {lake.table("silver", "movie")}"""
        ).fetchall()
        self.meta = {r[0]: r for r in rows}
        kw = model.vectorizers.get("kw")
        ids = model.movie_ids
        self.kw_matrix = (
            kw.transform([self.docs.get(int(i), {}).get("kw", []) for i in ids]) if kw else None
        )
        votes = np.array([self.meta.get(int(i), (0,) * 7)[5] or 0 for i in ids], dtype=float)
        avgs = np.array([self.meta.get(int(i), (0,) * 7)[4] or 0 for i in ids], dtype=float)
        # Bayesian shrinkage of vote_average toward the global mean (IMDb-style weighted rating).
        m, c = 200.0, float(np.average(avgs, weights=np.maximum(votes, 1)))
        self.quality = (votes * avgs + m * c) / (votes + m)

    def _theme(self, i: int, j: np.ndarray) -> np.ndarray:
        if self.kw_matrix is None:
            return np.zeros(len(j))
        return (self.kw_matrix[j] @ self.kw_matrix[i].T).toarray().ravel()

    def candidates(self, movie_id: int, k: int = 10, pool: int = 300) -> list[PairCandidate]:
        m = self.model
        i = m.index_of(movie_id)
        overall_all = m.embeddings @ m.embeddings[i]
        pool_idx = np.argsort(-overall_all, kind="stable")[1 : pool + 1]
        theme = self._theme(i, pool_idx)
        a = self.meta.get(movie_id)
        a_doc = self.docs.get(movie_id, {})
        out = []
        for jdx, th in zip(pool_idx, theme, strict=True):
            mid = int(m.movie_ids[jdx])
            b = self.meta.get(mid)
            if not a or not b:
                continue
            b_doc = self.docs.get(mid, {})
            same_dir = bool(set(a_doc.get("dir", [])) & set(b_doc.get("dir", [])))
            same_decade = a_doc.get("decade") == b_doc.get("decade")
            same_country = bool(set(a_doc.get("country", [])) & set(b_doc.get("country", [])))
            same_coll = a[6] is not None and a[6] == b[6]
            total_rt = (a[3] or 110) + (b[3] or 110)
            feats = {
                "theme": float(th),
                "overall": float(overall_all[jdx]),
                "contrast": (not same_dir) + (not same_decade) + (not same_country),
                "redundant": float(same_coll or (same_dir and same_decade)),
                "runtime": max(0.0, (total_rt - 220) / 60),
                "quality": float((self.quality[jdx] - 6.0) / 2.0),
            }
            score = sum(WEIGHTS[f] * v for f, v in feats.items())
            shared = sorted(set(a_doc.get("kw", [])) & set(b_doc.get("kw", [])))
            out.append(PairCandidate(mid, b[1], round(score, 4), feats, shared[:8]))
        out.sort(key=lambda c: (-c.score, c.movie_id))
        return out[:k]
