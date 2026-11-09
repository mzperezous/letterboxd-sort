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
from typing import NamedTuple

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


class FilmMeta(NamedTuple):
    movie_id: int
    title: str
    runtime_min: int | None
    vote_average: float | None
    vote_count: int | None
    collection_id: int | None


@dataclass
class PairCandidate:
    movie_id: int
    title: str
    score: float
    features: dict[str, float]
    shared_keywords: list[str]


def pair_features(
    a: FilmMeta,
    b: FilmMeta,
    a_doc: dict[str, list[str]],
    b_doc: dict[str, list[str]],
    *,
    theme: float,
    overall: float,
    quality_b: float,
) -> dict[str, float]:
    """Interpretable features for the pair (a, b). Also the input to a future learned model."""

    def shares(field: str) -> bool:
        return bool(set(a_doc.get(field, [])) & set(b_doc.get(field, [])))

    same_dir, same_decade, same_country = shares("dir"), shares("decade"), shares("country")
    same_coll = a.collection_id is not None and a.collection_id == b.collection_id
    total_runtime = (a.runtime_min or 110) + (b.runtime_min or 110)
    return {
        "theme": theme,
        "overall": overall,
        "contrast": float((not same_dir) + (not same_decade) + (not same_country)),
        "redundant": float(same_coll or (same_dir and same_decade)),
        "runtime": max(0.0, (total_runtime - 220) / 60),
        "quality": (quality_b - 6.0) / 2.0,
    }


def score(features: dict[str, float]) -> float:
    return sum(WEIGHTS[f] * v for f, v in features.items())


class DoubleFeatureScorer:
    def __init__(self, lake: Lake, model: ContentModel):
        self.model = model
        self.docs = build_documents(lake)
        rows = lake.query(
            f"""SELECT movie_id, title, runtime_min, vote_average, vote_count, collection_id
                FROM {lake.table("silver", "movie")}"""
        ).fetchall()
        self.meta = {r[0]: FilmMeta(*r) for r in rows}
        ids = [int(i) for i in model.movie_ids]
        kw = model.vectorizers.get("kw")
        self.kw_matrix = (
            kw.transform([self.docs.get(i, {}).get("kw", []) for i in ids]) if kw else None
        )
        votes = np.array(
            [(self.meta[i].vote_count or 0) if i in self.meta else 0 for i in ids], float
        )
        avgs = np.array(
            [(self.meta[i].vote_average or 0) if i in self.meta else 0 for i in ids], float
        )
        # Bayesian shrinkage toward the global mean (IMDb-style weighted rating): a 9.0 from 12
        # votes shouldn't beat an 8.2 from 20k.
        prior_n, prior_mean = 200.0, float(np.average(avgs, weights=np.maximum(votes, 1)))
        self.quality = (votes * avgs + prior_n * prior_mean) / (votes + prior_n)

    def _theme(self, i: int, j: np.ndarray) -> np.ndarray:
        """Keyword-only cosine (what the films are about), separate from the full embedding."""
        if self.kw_matrix is None:
            return np.zeros(len(j))
        return (self.kw_matrix[j] @ self.kw_matrix[i].T).toarray().ravel()

    def candidates(self, movie_id: int, k: int = 10, pool: int = 300) -> list[PairCandidate]:
        """Rank the `pool` nearest films by pair score. Raises KeyError for unknown films."""
        m = self.model
        i = m.index_of(movie_id)
        a = self.meta.get(movie_id)
        if a is None:
            raise KeyError(movie_id)
        overall = m.embeddings @ m.embeddings[i]
        pool_idx = np.argsort(-overall, kind="stable")[1 : pool + 1]
        theme = self._theme(i, pool_idx)
        a_doc = self.docs.get(movie_id, {})
        out = []
        for jdx, th in zip(pool_idx, theme, strict=True):
            b = self.meta.get(int(m.movie_ids[jdx]))
            if b is None:
                continue
            b_doc = self.docs.get(b.movie_id, {})
            feats = pair_features(
                a,
                b,
                a_doc,
                b_doc,
                theme=float(th),
                overall=float(overall[jdx]),
                quality_b=float(self.quality[jdx]),
            )
            shared = sorted(set(a_doc.get("kw", [])) & set(b_doc.get("kw", [])))[:8]
            out.append(PairCandidate(b.movie_id, b.title, round(score(feats), 4), feats, shared))
        out.sort(key=lambda c: (-c.score, c.movie_id))
        return out[:k]
