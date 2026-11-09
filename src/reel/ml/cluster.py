"""Clustering + cluster characterization (deterministic, no LLM).

1. Reduce embeddings with PCA (cheap, deterministic) to make density estimation easier.
2. HDBSCAN: density-based, finds a variable number of clusters, labels outliers -1 (noise)
   instead of forcing every film into a bucket. min_cluster_size is the main knob.
3. c-TF-IDF (class-based TF-IDF, the trick behind BERTopic): treat each cluster as one big
   document of tokens and rank tokens by how specific they are to that cluster vs. all others.
   This gives a readable profile such as  kw:heist  kw:double_cross  genre:crime  decade:1990s.

An LLM can later *name* clusters from these profiles, but the profiles themselves are cheap and
reproducible.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pyarrow as pa
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from reel.lake import Lake
from reel.ml.content import ContentModel, build_documents, load


def cluster_embeddings(
    emb: np.ndarray, min_cluster_size: int = 15, pca_dim: int = 32, seed: int = 0
) -> np.ndarray:
    X = emb
    if pca_dim and emb.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(emb)
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        cluster_selection_method="leaf",
        copy=True,
    ).fit_predict(X)


def ctfidf_profiles(
    labels: np.ndarray, token_docs: list[list[str]], top_n: int = 12
) -> dict[int, list[tuple[str, float]]]:
    clusters = sorted(set(labels) - {-1})
    counts = {c: Counter() for c in clusters}
    for lab, toks in zip(labels, token_docs, strict=True):
        if lab != -1:
            counts[lab].update(set(toks))
    total = Counter()
    for c in counts.values():
        total.update(c)
    avg_words = np.mean([sum(c.values()) for c in counts.values()]) if counts else 1.0
    out = {}
    for c, cnt in counts.items():
        size = sum(cnt.values()) or 1
        scored = [
            (tok, (n / size) * np.log(1 + avg_words / total[tok]))
            for tok, n in cnt.items()
            if n >= 2
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        out[c] = [(t, round(float(s), 5)) for t, s in scored[:top_n]]
    return out


def run(lake: Lake, min_cluster_size: int = 15, model: ContentModel | None = None) -> dict:
    model = model or load(lake)
    labels = cluster_embeddings(model.embeddings, min_cluster_size=min_cluster_size)
    docs = build_documents(lake)
    token_docs = [
        [f"{f}:{t}" for f, toks in docs.get(int(mid), {}).items() if f != "cast" for t in toks]
        for mid in model.movie_ids
    ]
    profiles = ctfidf_profiles(labels, token_docs)

    titles = dict(
        lake.query(f"SELECT movie_id, title FROM {lake.table('silver', 'movie')}").fetchall()
    )
    votes = dict(
        lake.query(f"SELECT movie_id, vote_count FROM {lake.table('silver', 'movie')}").fetchall()
    )
    sizes = Counter(labels.tolist())
    exemplars: dict[int, list[str]] = {}
    for c in profiles:
        members = [int(m) for m, lab in zip(model.movie_ids, labels, strict=True) if lab == c]
        members.sort(key=lambda m: -(votes.get(m) or 0))
        exemplars[c] = [titles.get(m, str(m)) for m in members[:8]]

    lake.write_table(
        "gold",
        "movie_cluster",
        pa.table(
            {
                "movie_id": model.movie_ids.tolist(),
                "cluster_id": labels.astype(int).tolist(),
                "model": [model.name] * len(labels),
            }
        ),
    )
    cids = sorted(profiles)
    lake.write_table(
        "gold",
        "cluster_profile",
        pa.table(
            {
                "cluster_id": cids,
                "size": [sizes[c] for c in cids],
                "top_terms": [[t for t, _ in profiles[c]] for c in cids],
                "term_scores": [[s for _, s in profiles[c]] for c in cids],
                "exemplars": [exemplars[c] for c in cids],
            }
        ),
    )
    return {
        "clusters": len(cids),
        "noise": int(sizes.get(-1, 0)),
        "clustered": int(len(labels) - sizes.get(-1, 0)),
    }
