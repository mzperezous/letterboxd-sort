"""API end-to-end against Postgres (skipped without REEL_TEST_PG_DSN)."""

import os

import pytest
from fastapi.testclient import TestClient

from reel.ml import cluster, content
from reel.serving import db, publish

DSN = os.environ.get("REEL_TEST_PG_DSN")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not DSN, reason="REEL_TEST_PG_DSN not set")]


@pytest.fixture
def client(seeded_lake, monkeypatch):
    import psycopg

    base, name = DSN.rsplit("/", 1)
    with psycopg.connect(f"{base}/postgres", autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{name}"')
    content.train(seeded_lake, dim=16, neighbors_k=5)
    cluster.run(seeded_lake, min_cluster_size=8)
    with db.connect(DSN) as conn:
        db.migrate(conn)
        publish.publish(seeded_lake, conn)

    from reel import config
    from reel.api import app as api

    monkeypatch.setenv("REEL_PG_DSN", DSN)
    config.get_settings.cache_clear()
    for fn in (api.lake, api.model, api.scorer):
        fn.cache_clear()
    monkeypatch.setattr(api, "lake", lambda: seeded_lake)
    with TestClient(api.app) as c:
        yield c
    config.get_settings.cache_clear()
    for fn in (api.model, api.scorer):
        fn.cache_clear()


def test_endpoints(client):
    assert client.get("/health").json() == {"ok": True, "movies": 75}

    hits = client.get("/movies/search", params={"q": "Heist Film 13"}).json()
    assert hits[0]["title"] == "Heist Film 130"

    sim = client.get("/movies/110/similar", params={"limit": 5}).json()
    assert len(sim) == 5 and all(s["title"].startswith("Space") for s in sim)

    df = client.get("/movies/100/double-features", params={"limit": 3}).json()
    assert len(df) == 3 and df[0]["movie_id"] != 101  # sequel is penalized

    mood = client.get("/mood", params={"q": "wedding in paris"}).json()
    assert "wedding" in mood["matched_tokens"]["kw"]
    assert all(r["title"].startswith("Romance") for r in mood["results"])

    assert client.get("/movies/424242/double-features").status_code == 404
    assert client.get("/clusters").json()


def test_comparisons_drive_recommendations(client):
    assert client.get("/users/me/recommendations").status_code == 409  # no data yet
    heist, romance = range(125, 150), range(150, 175)
    for winner, loser in zip(heist[:12], romance[:12], strict=True):
        r = client.post("/users/me/comparisons", json={"winner_id": winner, "loser_id": loser})
        assert r.status_code == 201
    recs = client.get("/users/me/recommendations", params={"limit": 5}).json()
    assert recs["pairs"] == 12
    assert all(r["title"].startswith("Heist") for r in recs["results"])
