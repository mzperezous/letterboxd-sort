"""Dagster definitions: the pipeline as a graph of software-defined assets.

    tmdb_id_export ─┐
    movielens_files ┼─► tmdb_seed ─┐
    letterboxd_rss ─┘              ├─► silver/* ─► content_model ─► clusters ─► postgres_catalog
    tmdb_changes (daily) ──────────┘

Assets call the same library functions as the CLI, so `reel ...` and Dagster never diverge.
Run locally with:  uv run dagster dev -m reel.pipelines.definitions
"""

import asyncio
from datetime import date, timedelta

import dagster as dg
from dagster import AssetExecutionContext

from reel.config import Settings
from reel.lake import Lake
from reel.ml import cluster, content
from reel.serving import db, publish
from reel.sources import letterboxd, movielens, tmdb
from reel.storage import store_from_settings
from reel.transform import SILVER, build_table


class ReelResource(dg.ConfigurableResource):
    """Builds Settings from REEL_* env vars; swap env per deployment, not code."""

    def settings(self) -> Settings:
        return Settings()

    def lake(self) -> Lake:
        s = self.settings()
        return Lake(store_from_settings(s), s)


def _meta(stats: dict) -> dict:
    return {
        k: dg.MetadataValue.json(v) if isinstance(v, dict | list) else v for k, v in stats.items()
    }


# ------------------------------------------------------------------------------ bronze
@dg.asset(group_name="bronze", kinds={"python"})
def tmdb_id_export(reel: ReelResource) -> dg.MaterializeResult:
    """Daily TMDB id export: new / deleted ids (raw file + id-level change log)."""

    async def run():
        async with tmdb.TmdbClient(reel.settings()) as c:
            return await tmdb.ingest_id_export(reel.lake(), c)

    return dg.MaterializeResult(metadata=_meta(asyncio.run(run())))


@dg.asset(group_name="bronze", kinds={"python"})
def movielens_files(reel: ReelResource) -> dg.MaterializeResult:
    """MovieLens dump (variant from REEL_MOVIELENS_VARIANT)."""
    return dg.MaterializeResult(metadata=_meta(movielens.ingest(reel.lake(), reel.settings())))


@dg.asset(group_name="bronze", kinds={"python"})
def letterboxd_rss(reel: ReelResource) -> dg.MaterializeResult:
    """Recent diary entries for REEL_LETTERBOXD_USERS."""
    s = reel.settings()
    users = [u.strip() for u in s.letterboxd_users.split(",") if u.strip()]
    lake = reel.lake()
    return dg.MaterializeResult(
        metadata=_meta({u: letterboxd.ingest_rss(lake, s, u) for u in users})
    )


@dg.asset(group_name="bronze", deps=[tmdb_id_export, letterboxd_rss], kinds={"python"})
def tmdb_seed(reel: ReelResource) -> dg.MaterializeResult:
    """Fetch details for catalog films not yet in bronze (top voted + popular + any films
    referenced by MovieLens / Letterboxd data already in the lake)."""
    s, lake = reel.settings(), reel.lake()
    for name in ("ml_movie", "lb_rss_entry"):  # so referenced films are included
        build_table(lake, name)
    extra: list[int] = []
    for name in ("ml_movie", "lb_rss_entry"):
        if lake.has_table("silver", name):
            extra += [
                r[0]
                for r in lake.query(
                    f"SELECT movie_id FROM {lake.table('silver', name)} WHERE movie_id IS NOT NULL"
                ).fetchall()
            ]

    async def run():
        async with tmdb.TmdbClient(s) as c:
            voted = await c.discover_ids(pages=(s.seed_top_voted + 19) // 20)
            ids = tmdb.select_seed_ids(lake, s.seed_popular, voted[: s.seed_top_voted] + extra)
            return await tmdb.ingest_movies(lake, c, ids, only_missing=True)

    return dg.MaterializeResult(metadata=_meta(asyncio.run(run())))


daily = dg.DailyPartitionsDefinition(start_date="2026-11-01", end_offset=0)


@dg.asset(group_name="bronze", partitions_def=daily, deps=[tmdb_seed], kinds={"python"})
def tmdb_changes(context: AssetExecutionContext, reel: ReelResource) -> dg.MaterializeResult:
    """CDC: re-fetch catalog films TMDB reports changed on this partition's day."""
    day = date.fromisoformat(context.partition_key)

    async def run():
        async with tmdb.TmdbClient(reel.settings()) as c:
            return await tmdb.ingest_changes(reel.lake(), c, day, day + timedelta(days=1))

    return dg.MaterializeResult(metadata=_meta(asyncio.run(run())))


# ------------------------------------------------------------------------------ silver
SOURCE_DEPS = {
    "tmdb_movie_history": [tmdb_seed, tmdb_changes],
    "tmdb_id_history": [tmdb_id_export],
    "ml_movie": [movielens_files],
    "ml_rating": [movielens_files],
    "ml_tag": [movielens_files],
    "ml_genome": [movielens_files],
    "lb_rating": [],
    "lb_watched": [],
    "lb_rss_entry": [letterboxd_rss],
}


def _silver_asset(name: str):
    spec = SILVER[name]
    deps = [dg.AssetKey(["silver", d]) for d in spec.deps] + SOURCE_DEPS.get(name, [])

    @dg.asset(
        name=name,
        key_prefix="silver",
        group_name="silver",
        deps=deps,
        description=spec.description,
        kinds={"duckdb"},
    )
    def _asset(reel: ReelResource) -> dg.MaterializeResult:
        rows = build_table(reel.lake(), name)
        return dg.MaterializeResult(metadata={"rows": rows if rows is not None else -1})

    return _asset


silver_assets = [_silver_asset(n) for n in SILVER]


# ------------------------------------------------------------------------------ gold
@dg.asset(
    group_name="gold",
    deps=[
        dg.AssetKey(["silver", n])
        for n in (
            "movie",
            "movie_genre",
            "movie_keyword",
            "movie_cast",
            "movie_crew",
            "movie_country",
        )
    ],
    kinds={"sklearn"},
)
def content_model(reel: ReelResource) -> dg.MaterializeResult:
    """TF-IDF + LSA embeddings and exact kNN neighbors."""
    return dg.MaterializeResult(metadata=_meta(content.train(reel.lake())))


@dg.asset(group_name="gold", deps=[content_model], kinds={"sklearn"})
def clusters(reel: ReelResource) -> dg.MaterializeResult:
    """HDBSCAN clusters + c-TF-IDF profiles."""
    return dg.MaterializeResult(metadata=_meta(cluster.run(reel.lake())))


@dg.asset(group_name="serving", deps=[content_model, clusters], kinds={"postgres"})
def postgres_catalog(reel: ReelResource) -> dg.MaterializeResult:
    """Sync catalog + model outputs into Postgres for the API."""
    with db.connect(reel.settings().pg_dsn) as conn:
        db.migrate(conn)
        stats = publish.publish(reel.lake(), conn)
    return dg.MaterializeResult(metadata=_meta(stats))


# ------------------------------------------------------------------------------ jobs
everything_but_changes = dg.AssetSelection.all() - dg.AssetSelection.assets(tmdb_changes)
bootstrap_job = dg.define_asset_job("bootstrap", selection=everything_but_changes)
daily_refresh_job = dg.define_asset_job(
    "daily_refresh",
    selection=dg.AssetSelection.assets(tmdb_changes, tmdb_id_export, letterboxd_rss)
    | dg.AssetSelection.assets(tmdb_changes).downstream(),
    partitions_def=daily,
)
daily_refresh_schedule = dg.build_schedule_from_partitioned_job(
    daily_refresh_job, hour_of_day=9, minute_of_hour=17
)

defs = dg.Definitions(
    assets=[
        tmdb_id_export,
        movielens_files,
        letterboxd_rss,
        tmdb_seed,
        tmdb_changes,
        *silver_assets,
        content_model,
        clusters,
        postgres_catalog,
    ],
    jobs=[bootstrap_job, daily_refresh_job],
    schedules=[daily_refresh_schedule],
    resources={"reel": ReelResource()},
)
