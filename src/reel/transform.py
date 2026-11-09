"""Silver layer: normalized tables built from bronze with DuckDB SQL.

Every silver table is a pure function of bronze/files, rebuilt in full. At this data size (tens
to hundreds of thousands of films) a full rebuild takes seconds and removes a whole class of
incremental-merge bugs. When it stops being cheap, the same SQL can be run per partition.

Tables are registered in SILVER with their dependencies, so the CLI and Dagster share one graph.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from reel.lake import Lake, scd2_sql
from reel.sources import movielens

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SilverTable:
    name: str
    build: Callable[[Lake], str | None]  # returns SQL, or None if inputs are absent
    deps: tuple[str, ...] = ()
    description: str = ""


SILVER: dict[str, SilverTable] = {}


def silver(name: str, deps: tuple[str, ...] = (), description: str = ""):
    def deco(fn):
        SILVER[name] = SilverTable(name, fn, deps, description or (fn.__doc__ or "").strip())
        return fn

    return deco


# ---------------------------------------------------------------------------- TMDB


@silver("tmdb_movie_history")
def _tmdb_movie_history(lake: Lake) -> str | None:
    """SCD2 history of every TMDB movie payload we've fetched."""
    if not lake.bronze_exists("tmdb", "movie"):
        return None
    return f"SELECT CAST(key AS BIGINT) AS movie_id, * EXCLUDE (key) FROM ({scd2_sql(lake.bronze('tmdb', 'movie'))})"


def _current_tmdb(lake: Lake) -> str:
    return f"""(SELECT movie_id, payload::JSON AS j FROM {lake.table("silver", "tmdb_movie_history")}
               WHERE is_current AND NOT is_deleted)"""


@silver("movie", deps=("tmdb_movie_history",))
def _movie(lake: Lake) -> str:
    """One row per live movie: core attributes (current version)."""
    return f"""
    SELECT
        movie_id,
        j->>'title' AS title,
        j->>'original_title' AS original_title,
        j->>'original_language' AS original_language,
        TRY_CAST(nullif(j->>'release_date', '') AS DATE) AS release_date,
        year(TRY_CAST(nullif(j->>'release_date', '') AS DATE)) AS release_year,
        nullif(CAST(j->>'runtime' AS INTEGER), 0) AS runtime_min,
        j->>'overview' AS overview,
        nullif(j->>'tagline', '') AS tagline,
        j->>'status' AS status,
        CAST(j->>'vote_average' AS DOUBLE) AS vote_average,
        CAST(j->>'vote_count' AS INTEGER) AS vote_count,
        nullif(CAST(j->>'budget' AS BIGINT), 0) AS budget,
        nullif(CAST(j->>'revenue' AS BIGINT), 0) AS revenue,
        j->>'poster_path' AS poster_path,
        nullif(j->>'imdb_id', '') AS imdb_id,
        nullif(j->'external_ids'->>'wikidata_id', '') AS wikidata_id,
        CAST(j->'belongs_to_collection'->>'id' AS BIGINT) AS collection_id,
        j->'belongs_to_collection'->>'name' AS collection_name
    FROM {_current_tmdb(lake)}
    """


def _unnest_sql(lake: Lake, json_expr: str, struct: str, select: str) -> str:
    """Explode a JSON array on the current TMDB payload into rows."""
    return f"""
    WITH x AS (
        SELECT movie_id, unnest(from_json({json_expr}, '{struct}')) AS e
        FROM {_current_tmdb(lake)}
    )
    SELECT movie_id, {select} FROM x
    """


@silver("movie_genre", deps=("tmdb_movie_history",))
def _movie_genre(lake: Lake) -> str:
    return _unnest_sql(
        lake,
        "j->'genres'",
        '[{"id":"BIGINT","name":"VARCHAR"}]',
        "e.id AS genre_id, e.name AS genre",
    )


@silver("movie_keyword", deps=("tmdb_movie_history",))
def _movie_keyword(lake: Lake) -> str:
    return _unnest_sql(
        lake,
        "j->'keywords'->'keywords'",
        '[{"id":"BIGINT","name":"VARCHAR"}]',
        "e.id AS keyword_id, e.name AS keyword",
    )


@silver("movie_cast", deps=("tmdb_movie_history",))
def _movie_cast(lake: Lake) -> str:
    return _unnest_sql(
        lake,
        "j->'credits'->'cast'",
        '[{"id":"BIGINT","name":"VARCHAR","character":"VARCHAR","order":"INTEGER"}]',
        'e.id AS person_id, e.name AS name, e.character AS character, e."order" AS billing_order',
    )


@silver("movie_crew", deps=("tmdb_movie_history",))
def _movie_crew(lake: Lake) -> str:
    return _unnest_sql(
        lake,
        "j->'credits'->'crew'",
        '[{"id":"BIGINT","name":"VARCHAR","job":"VARCHAR","department":"VARCHAR"}]',
        "e.id AS person_id, e.name AS name, e.job AS job, e.department AS department",
    )


@silver("movie_country", deps=("tmdb_movie_history",))
def _movie_country(lake: Lake) -> str:
    return _unnest_sql(
        lake,
        "j->'production_countries'",
        '[{"iso_3166_1":"VARCHAR","name":"VARCHAR"}]',
        "e.iso_3166_1 AS country_code, e.name AS country",
    )


@silver("tmdb_id_history")
def _tmdb_id_history(lake: Lake) -> str | None:
    """SCD2 of TMDB's daily id export: when ids appeared, were retitled, or deleted."""
    if not lake.bronze_exists("tmdb", "id_export"):
        return None
    return f"SELECT CAST(key AS BIGINT) AS movie_id, * EXCLUDE (key) FROM ({scd2_sql(lake.bronze('tmdb', 'id_export'))})"


# ---------------------------------------------------------------------------- MovieLens


def _ml_variant(lake: Lake) -> str | None:
    """Prefer the richest ingested variant (genome > size)."""
    variants = movielens.ingested_variants(lake)
    for v in ("ml-25m", "ml-32m", "ml-latest", "ml-latest-small"):
        if v in variants:
            return v
    return variants[0] if variants else None


def _ml_csv(lake: Lake, name: str) -> str | None:
    v = _ml_variant(lake)
    uri = movielens.csv_uri(lake, v, name) if v else None
    return f"read_csv('{uri}', header = true)" if uri else None


@silver("ml_movie")
def _ml_movie(lake: Lake) -> str | None:
    """MovieLens movies joined to their TMDB/IMDb ids."""
    movies, links = _ml_csv(lake, "movies.csv"), _ml_csv(lake, "links.csv")
    if not movies or not links:
        return None
    return f"""
    SELECT m.movieId AS ml_movie_id,
           m.title AS ml_title,
           TRY_CAST(regexp_extract(m.title, '\\((\\d{{4}})\\)\\s*$', 1) AS INTEGER) AS ml_year,
           string_split(m.genres, '|') AS ml_genres,
           CAST(l.tmdbId AS BIGINT) AS movie_id,
           'tt' || lpad(CAST(l.imdbId AS VARCHAR), 7, '0') AS imdb_id
    FROM {movies} m LEFT JOIN {links} l USING (movieId)
    """


@silver("ml_rating")
def _ml_rating(lake: Lake) -> str | None:
    """MovieLens ratings (0.5..5) for collaborative filtering."""
    ratings = _ml_csv(lake, "ratings.csv")
    if not ratings:
        return None
    return f"""
    SELECT userId AS ml_user_id, movieId AS ml_movie_id, CAST(rating AS FLOAT) AS rating,
           to_timestamp(timestamp) AS rated_at
    FROM {ratings}
    """


@silver("ml_tag")
def _ml_tag(lake: Lake) -> str | None:
    """Free-text user tags ("atmospheric", "slow burn", ...)."""
    tags = _ml_csv(lake, "tags.csv")
    if not tags:
        return None
    return f"""
    SELECT userId AS ml_user_id, movieId AS ml_movie_id, lower(trim(CAST(tag AS VARCHAR))) AS tag,
           to_timestamp(timestamp) AS tagged_at
    FROM {tags} WHERE tag IS NOT NULL
    """


@silver("ml_genome")
def _ml_genome(lake: Lake) -> str | None:
    """Tag Genome: dense relevance (0..1) of 1,128 tags per movie. Only in ml-25m-style dumps."""
    scores, tags = _ml_csv(lake, "genome-scores.csv"), _ml_csv(lake, "genome-tags.csv")
    if not scores or not tags:
        return None
    return f"""
    SELECT s.movieId AS ml_movie_id, s.tagId AS tag_id, t.tag, CAST(s.relevance AS FLOAT) AS relevance
    FROM {scores} s JOIN {tags} t USING (tagId)
    """


# ---------------------------------------------------------------------------- Letterboxd


def _lb_latest(lake: Lake, entity: str) -> str | None:
    if not lake.bronze_exists("letterboxd", entity):
        return None
    return f"""(SELECT key, payload::JSON AS j FROM ({scd2_sql(lake.bronze("letterboxd", entity))})
               WHERE is_current AND NOT is_deleted)"""


@silver("lb_rating")
def _lb_rating(lake: Lake) -> str | None:
    """Current star rating per user x film from Letterboxd exports."""
    src = _lb_latest(lake, "export_rating")
    if not src:
        return None
    return f"""
    SELECT j->>'username' AS username, j->>'Name' AS film_title,
           TRY_CAST(j->>'Year' AS INTEGER) AS film_year, j->>'Letterboxd URI' AS letterboxd_uri,
           TRY_CAST(j->>'Rating' AS DOUBLE) AS rating, TRY_CAST(j->>'Date' AS DATE) AS rated_on
    FROM {src}
    """


@silver("lb_watched")
def _lb_watched(lake: Lake) -> str | None:
    """Films each user has logged as watched."""
    src = _lb_latest(lake, "export_watched")
    if not src:
        return None
    return f"""
    SELECT j->>'username' AS username, j->>'Name' AS film_title,
           TRY_CAST(j->>'Year' AS INTEGER) AS film_year, j->>'Letterboxd URI' AS letterboxd_uri,
           TRY_CAST(j->>'Date' AS DATE) AS logged_on
    FROM {src}
    """


@silver("lb_rss_entry")
def _lb_rss_entry(lake: Lake) -> str | None:
    """Diary entries from RSS: have TMDB ids and review text."""
    src = _lb_latest(lake, "rss_entry")
    if not src:
        return None
    return f"""
    SELECT j->>'username' AS username, j->>'link' AS link, j->>'film_title' AS film_title,
           CAST(j->>'film_year' AS INTEGER) AS film_year, CAST(j->>'tmdb_id' AS BIGINT) AS movie_id,
           TRY_CAST(j->>'watched_date' AS DATE) AS watched_on, CAST(j->>'rating' AS DOUBLE) AS rating,
           CAST(j->>'liked' AS BOOLEAN) AS liked, CAST(j->>'rewatch' AS BOOLEAN) AS rewatch,
           j->>'review_text' AS review_text
    FROM {src}
    """


# ---------------------------------------------------------------------------- crosswalk


@silver("movie_xref", deps=("movie", "ml_movie"))
def _movie_xref(lake: Lake) -> str | None:
    """Entity resolution hub: one row per TMDB id with every external id we know."""
    parts = []
    if lake.has_table("silver", "movie"):
        parts.append(
            f"SELECT movie_id, imdb_id, wikidata_id, NULL::BIGINT AS ml_movie_id FROM {lake.table('silver', 'movie')}"
        )
    if lake.has_table("silver", "ml_movie"):
        parts.append(
            f"SELECT movie_id, imdb_id, NULL AS wikidata_id, ml_movie_id FROM {lake.table('silver', 'ml_movie')} WHERE movie_id IS NOT NULL"
        )
    if not parts:
        return None
    return f"""
    SELECT movie_id, any_value(imdb_id) AS imdb_id, any_value(wikidata_id) AS wikidata_id,
           any_value(ml_movie_id) AS ml_movie_id
    FROM ({" UNION ALL ".join(parts)})
    GROUP BY movie_id
    """


# ---------------------------------------------------------------------------- runner


def _topo(names: list[str]) -> list[str]:
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for d in SILVER[n].deps:
            visit(d)
        order.append(n)

    for n in names:
        visit(n)
    return order


def build_table(lake: Lake, name: str) -> int | None:
    """Build one silver table. Returns row count, or None if its inputs don't exist yet
    (a builder returns None, or references a table that hasn't been built)."""
    try:
        sql = SILVER[name].build(lake)
    except LookupError as e:
        log.info("skip %s: %s", name, e)
        return None
    if sql is None:
        log.info("skip %s: inputs absent", name)
        return None
    n = lake.write_table("silver", name, lake.sql(sql))
    log.info("silver.%s: %d rows", name, n)
    return n


def build_silver(lake: Lake, only: list[str] | None = None) -> dict[str, int | None]:
    return {name: build_table(lake, name) for name in _topo(only or list(SILVER))}
