"""Publish lake (silver/gold) -> Postgres `catalog` schema for the API to serve.

The lake is the source of truth for catalog data; Postgres is a disposable read model. Each
publish is one transaction, so the API never sees a half-synced catalog.
"""

from __future__ import annotations

import psycopg

from reel.lake import Lake
from reel.serving.db import sync_table


def _vec(v) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def publish(lake: Lake, conn: psycopg.Connection) -> dict:
    t = lambda n: lake.table("silver", n)  # noqa: E731
    out: dict[str, dict] = {}
    movies = lake.query(
        f"""
        WITH g AS (SELECT movie_id, list(genre ORDER BY genre) AS genres FROM {t("movie_genre")} GROUP BY 1),
             k AS (SELECT movie_id, list(keyword ORDER BY keyword) AS keywords FROM {t("movie_keyword")} GROUP BY 1),
             d AS (SELECT movie_id, list(DISTINCT name) AS directors FROM {t("movie_crew")}
                   WHERE job = 'Director' GROUP BY 1)
        SELECT m.movie_id, m.title, m.original_title, m.release_year, m.runtime_min, m.overview,
               m.poster_path, m.vote_average, m.vote_count, m.imdb_id,
               coalesce(g.genres, []), coalesce(k.keywords, []), coalesce(d.directors, [])
        FROM {t("movie")} m LEFT JOIN g USING (movie_id) LEFT JOIN k USING (movie_id)
        LEFT JOIN d USING (movie_id)
        WHERE m.title IS NOT NULL
        """
    ).fetchall()
    out["movie"] = sync_table(
        conn,
        "catalog.movie",
        [
            "movie_id",
            "title",
            "original_title",
            "release_year",
            "runtime_min",
            "overview",
            "poster_path",
            "vote_average",
            "vote_count",
            "imdb_id",
            "genres",
            "keywords",
            "directors",
        ],
        ["movie_id"],
        movies,
    )
    live = {r[0] for r in movies}

    if lake.has_table("gold", "movie_embedding"):
        rows = lake.query(
            f"SELECT movie_id, model, embedding FROM {lake.table('gold', 'movie_embedding')}"
        ).fetchall()
        models = {r[1] for r in rows}
        for model in models:
            out[f"embedding:{model}"] = sync_table(
                conn,
                "catalog.movie_embedding",
                ["movie_id", "model", "embedding"],
                ["movie_id", "model"],
                ((m, md, _vec(e)) for m, md, e in rows if md == model and m in live),
                scope=("model", model),
            )
    if lake.has_table("gold", "movie_neighbor"):
        rows = lake.query(
            f"SELECT movie_id, model, rank, neighbor_id, score FROM {lake.table('gold', 'movie_neighbor')} WHERE rank <= 30"
        ).fetchall()
        out["neighbor"] = sync_table(
            conn,
            "catalog.movie_neighbor",
            ["movie_id", "model", "rank", "neighbor_id", "score"],
            ["movie_id", "model", "rank"],
            rows,
        )
    if lake.has_table("gold", "movie_cluster"):
        rows = lake.query(
            f"SELECT movie_id, model, cluster_id FROM {lake.table('gold', 'movie_cluster')}"
        ).fetchall()
        out["cluster"] = sync_table(
            conn,
            "catalog.movie_cluster",
            ["movie_id", "model", "cluster_id"],
            ["movie_id", "model"],
            rows,
        )
        rows = lake.query(
            f"SELECT cluster_id, size, top_terms, exemplars FROM {lake.table('gold', 'cluster_profile')}"
        ).fetchall()
        out["cluster_profile"] = sync_table(
            conn,
            "catalog.cluster_profile",
            ["cluster_id", "size", "top_terms", "exemplars"],
            ["cluster_id"],
            rows,
        )
    conn.commit()
    return out
