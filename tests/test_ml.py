import numpy as np

from reel.ml import cluster, content, taste
from reel.ml.pairs import DoubleFeatureScorer


def theme_of(lake, mid):
    title = lake.query(
        f"SELECT title FROM {lake.table('silver', 'movie')} WHERE movie_id = ?", params=[mid]
    ).fetchone()[0]
    return title.split()[0].lower()


def test_content_model_is_deterministic_and_thematic(seeded_lake):
    docs = content.build_documents(seeded_lake)
    a = content.ContentModel.fit(docs, dim=16)
    b = content.ContentModel.fit(docs, dim=16)
    np.testing.assert_allclose(a.embeddings, b.embeddings, atol=1e-6)
    assert np.allclose(np.linalg.norm(a.embeddings, axis=1), 1, atol=1e-5)

    # Nearest neighbors of a space film are overwhelmingly space films.
    sims = a.similar(110, k=10)
    assert sum(theme_of(seeded_lake, m) == "space" for m, _ in sims) >= 9


def test_train_writes_artifacts_and_roundtrips(seeded_lake):
    stats = content.train(seeded_lake, dim=16, neighbors_k=5)
    assert stats["movies"] == 75 and stats["dim"] == 16
    m = content.load(seeded_lake)
    nn = seeded_lake.query(
        f"SELECT count(*), max(rank) FROM {seeded_lake.table('gold', 'movie_neighbor')}"
    ).fetchone()
    assert nn == (75 * 5, 5)
    # neighbors table agrees with the in-memory ranking
    top = seeded_lake.query(
        f"SELECT neighbor_id FROM {seeded_lake.table('gold', 'movie_neighbor')} WHERE movie_id = 120 ORDER BY rank"
    ).fetchall()
    assert [r[0] for r in top] == [mid for mid, _ in m.similar(120, k=5)]


def test_mood_query_folding(seeded_lake):
    m = content.ContentModel.fit(content.build_documents(seeded_lake), dim=16)
    hits, toks = m.search_text("a heist with a double cross", k=10)
    assert "double_cross" in toks["kw"] and "heist" in toks["kw"]
    assert all(theme_of(seeded_lake, mid) == "heist" for mid, _ in hits)
    assert m.search_text("zzzz qqqq", k=5)[0] == []


def test_clusters_recover_themes(seeded_lake):
    m = content.ContentModel.fit(content.build_documents(seeded_lake), dim=16)
    labels = cluster.cluster_embeddings(m.embeddings, min_cluster_size=8, pca_dim=8)
    # Each discovered cluster should be (nearly) pure in theme.
    for c in set(labels) - {-1}:
        themes = [
            theme_of(seeded_lake, int(mid))
            for mid, lab in zip(m.movie_ids, labels, strict=True)
            if lab == c
        ]
        assert max(themes.count(t) for t in set(themes)) / len(themes) >= 0.9


def test_cluster_run_writes_profiles(seeded_lake):
    content.train(seeded_lake, dim=16, neighbors_k=5)
    out = cluster.run(seeded_lake, min_cluster_size=8)
    assert out["clusters"] >= 2
    terms = seeded_lake.query(
        f"SELECT top_terms FROM {seeded_lake.table('gold', 'cluster_profile')}"
    ).fetchall()
    assert all(t[0] for t in terms)


def test_double_features_penalize_sequels(seeded_lake):
    m = content.ContentModel.fit(content.build_documents(seeded_lake), dim=16)
    scorer = DoubleFeatureScorer(seeded_lake, m)
    cands = scorer.candidates(100, k=74, pool=74)
    by_id = {c.movie_id: c for c in cands}
    assert by_id[101].features["redundant"] == 1.0  # same collection
    assert cands[0].movie_id != 101
    assert all(theme_of(seeded_lake, c.movie_id) == "space" for c in cands[:5])


def test_taste_model_learns_preference(seeded_lake):
    m = content.ContentModel.fit(content.build_documents(seeded_lake), dim=16)
    ids = [int(i) for i in m.movie_ids]
    # User loves heist films, dislikes romance.
    ratings = {
        i: (
            5.0
            if theme_of(seeded_lake, i) == "heist"
            else 1.0
            if theme_of(seeded_lake, i) == "romance"
            else 3.0
        )
        for i in ids
    }
    train_ids = ids[::2]
    test_ids = ids[1::2]
    t = taste.fit(m, taste.pairs_from_ratings({i: ratings[i] for i in train_ids}))
    acc = taste.pairwise_accuracy(t, m, taste.pairs_from_ratings({i: ratings[i] for i in test_ids}))
    assert acc > 0.9
