"""Integration tests against a real Postgres+pgvector. Run `make up` and set
REEL_TEST_PG_DSN=postgresql://reel:reel@localhost:5433/reel_test (database is recreated)."""

import os

import psycopg
import pytest

from reel.ml import cluster, content
from reel.serving import db, publish

DSN = os.environ.get("REEL_TEST_PG_DSN")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not DSN, reason="REEL_TEST_PG_DSN not set")]


@pytest.fixture
def conn():
    base, name = DSN.rsplit("/", 1)
    with psycopg.connect(f"{base}/postgres", autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{name}"')
    with db.connect(DSN) as c:
        yield c


def test_migrate_is_idempotent(conn):
    assert db.migrate(conn) == ["001_init.sql"]
    assert db.migrate(conn) == []


def test_publish_and_vector_search(conn, seeded_lake):
    db.migrate(conn)
    content.train(seeded_lake, dim=16, neighbors_k=5)
    cluster.run(seeded_lake, min_cluster_size=8)
    out = publish.publish(seeded_lake, conn)
    assert out["movie"]["rows"] == 75

    # pgvector ranking agrees with the in-memory model
    rows = conn.execute(
        """SELECT e.movie_id FROM catalog.movie_embedding e, catalog.movie_embedding q
           WHERE q.movie_id = 130 AND e.movie_id <> 130 AND e.model = q.model
           ORDER BY e.embedding <=> q.embedding LIMIT 5"""
    ).fetchall()
    m = content.load(seeded_lake)
    assert [r[0] for r in rows] == [mid for mid, _ in m.similar(130, k=5)]

    # Re-publish after a movie disappears from the lake deletes it (and cascades embeddings).
    seeded_lake.write_table(
        "silver",
        "movie",
        seeded_lake.sql(
            f"SELECT * FROM {seeded_lake.table('silver', 'movie')} WHERE movie_id <> 130"
        ),
    )
    out = publish.publish(seeded_lake, conn)
    assert out["movie"]["deleted"] == 1
    assert conn.execute(
        "SELECT count(*) FROM catalog.movie_embedding WHERE movie_id = 130"
    ).fetchone() == (0,)
