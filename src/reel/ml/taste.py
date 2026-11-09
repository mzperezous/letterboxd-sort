"""Personal taste from pairwise preferences (Bradley-Terry over film embeddings).

Model: P(user prefers a over b) = sigmoid(w . (x_a - x_b)), where x is a film embedding and w
is a per-user taste vector. Fitting w is plain L2-regularized logistic regression on embedding
*differences*, deterministic and fast. Score any unseen film with w . x.

Training pairs come from:
  - explicit comparisons in the app ("better / worse", Beli-style ranking), the best signal
  - star ratings, expanded into pairs between films with different ratings

Evaluation: pairwise accuracy on held-out comparisons (does w order unseen pairs correctly?).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression

from reel.ml.content import ContentModel


@dataclass
class TasteModel:
    w: np.ndarray
    n_pairs: int

    def score(self, emb: np.ndarray) -> np.ndarray:
        return emb @ self.w


def pairs_from_ratings(
    ratings: dict[int, float], max_pairs: int = 20000, seed: int = 0
) -> list[tuple[int, int]]:
    """(winner, loser) for every pair of films with different ratings (sampled if huge)."""
    pairs = [
        (a, b) if ra > rb else (b, a)
        for (a, ra), (b, rb) in combinations(sorted(ratings.items()), 2)
        if ra != rb
    ]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), max_pairs, replace=False))]
    return pairs


def fit(model: ContentModel, pairs: Iterable[tuple[int, int]], C: float = 1.0) -> TasteModel:
    idx = {int(m): i for i, m in enumerate(model.movie_ids)}
    diffs = [
        model.embeddings[idx[a]] - model.embeddings[idx[b]]
        for a, b in pairs
        if a in idx and b in idx
    ]
    if len(diffs) < 2:
        raise ValueError("need at least 2 usable preference pairs")
    D = np.vstack(diffs)
    # Symmetrize: each pair as (a-b, 1) and (b-a, 0) so no intercept is needed.
    X = np.vstack([D, -D])
    y = np.r_[np.ones(len(D)), np.zeros(len(D))]
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=1000)
    clf.fit(X, y)
    return TasteModel(clf.coef_[0].astype(np.float32), len(D))


def pairwise_accuracy(
    taste: TasteModel, model: ContentModel, pairs: Iterable[tuple[int, int]]
) -> float:
    idx = {int(m): i for i, m in enumerate(model.movie_ids)}
    s = taste.score(model.embeddings)
    usable = [(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx]
    if not usable:
        return float("nan")
    return float(np.mean([s[a] > s[b] for a, b in usable]))
