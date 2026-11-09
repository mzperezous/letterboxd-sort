"""`reel` command line. Thin wrappers over library functions, which Dagster assets also call."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import typer

from reel.config import get_settings
from reel.lake import Lake
from reel.storage import store_from_settings

app = typer.Typer(no_args_is_help=True, help="Film data platform CLI.")
tmdb_app = typer.Typer(no_args_is_help=True, help="TMDB ingestion.")
ml_app = typer.Typer(no_args_is_help=True, help="Models: embeddings, clusters, similarity.")
db_app = typer.Typer(no_args_is_help=True, help="Serving database (Postgres).")
lb_app = typer.Typer(no_args_is_help=True, help="Letterboxd ingestion.")
app.add_typer(tmdb_app, name="tmdb")
app.add_typer(ml_app, name="ml")
app.add_typer(db_app, name="db")
app.add_typer(lb_app, name="letterboxd")


def _lake() -> Lake:
    s = get_settings()
    return Lake(store_from_settings(s), s)


def _print(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.callback()
def main(verbose: bool = typer.Option(False, "-v", "--verbose")):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------------------- TMDB
@tmdb_app.command("export")
def tmdb_export(day: str = typer.Option(None, help="YYYY-MM-DD; default latest available")):
    """Ingest TMDB's daily movie id export (new / deleted ids)."""
    from reel.sources import tmdb

    async def run():
        async with tmdb.TmdbClient(get_settings()) as c:
            return await tmdb.ingest_id_export(_lake(), c, date.fromisoformat(day) if day else None)

    _print(asyncio.run(run()))


@tmdb_app.command("seed")
def tmdb_seed(
    limit: int = typer.Option(2000, help="Top-N ids by current popularity (latest export)"),
    top_voted: int = typer.Option(10000, help="Top-N ids by all-time vote count (the canon)"),
    include_movielens: bool = typer.Option(True, help="Also fetch every MovieLens-linked film"),
    include_letterboxd: bool = typer.Option(True, help="Also fetch films from Letterboxd RSS"),
):
    """Fetch details for the seed catalog (skips ids already fetched)."""
    from reel.sources import tmdb

    lake = _lake()
    extra: list[int] = []
    if include_movielens and lake.has_table("silver", "ml_movie"):
        extra += [
            r[0]
            for r in lake.query(
                f"SELECT movie_id FROM {lake.table('silver', 'ml_movie')} WHERE movie_id IS NOT NULL"
            ).fetchall()
        ]
    if include_letterboxd and lake.has_table("silver", "lb_rss_entry"):
        extra += [
            r[0]
            for r in lake.query(
                f"SELECT movie_id FROM {lake.table('silver', 'lb_rss_entry')} WHERE movie_id IS NOT NULL"
            ).fetchall()
        ]

    async def run():
        async with tmdb.TmdbClient(get_settings()) as c:
            voted = await c.discover_ids(pages=(top_voted + 19) // 20) if top_voted else []
            ids = tmdb.select_seed_ids(lake, limit, voted[:top_voted] + extra)
            return await tmdb.ingest_movies(lake, c, ids, only_missing=True)

    _print(asyncio.run(run()))


@tmdb_app.command("fetch")
def tmdb_fetch(ids: list[int]):
    """Fetch (or refresh) specific TMDB movie ids."""
    from reel.sources import tmdb

    async def run():
        async with tmdb.TmdbClient(get_settings()) as c:
            return await tmdb.ingest_movies(_lake(), c, ids)

    _print(asyncio.run(run()))


@tmdb_app.command("changes")
def tmdb_changes(
    start: str = typer.Option(None, help="YYYY-MM-DD; default yesterday"),
    end: str = typer.Option(None, help="YYYY-MM-DD; default today"),
):
    """CDC: re-fetch catalog movies TMDB reports as changed."""
    from reel.sources import tmdb

    today = datetime.now(UTC).date()
    s = date.fromisoformat(start) if start else today - timedelta(days=1)
    e = date.fromisoformat(end) if end else today

    async def run():
        async with tmdb.TmdbClient(get_settings()) as c:
            return await tmdb.ingest_changes(_lake(), c, s, e)

    _print(asyncio.run(run()))


# ------------------------------------------------------------------------------- MovieLens
@app.command("movielens")
def movielens_ingest(
    variant: str = typer.Option(None, help="ml-latest-small | ml-25m | ml-32m"),
    zip_path: Path = typer.Option(None, "--zip", help="Use a locally downloaded zip"),
):
    """Land a MovieLens dump in the lake."""
    from reel.sources import movielens

    _print(movielens.ingest(_lake(), get_settings(), variant, zip_path))


# ------------------------------------------------------------------------------- Letterboxd
@lb_app.command("rss")
def lb_rss(usernames: list[str]):
    """Ingest recent diary entries (with TMDB ids and review text) from public RSS."""
    from reel.sources import letterboxd

    lake, s = _lake(), get_settings()
    _print({u: letterboxd.ingest_rss(lake, s, u) for u in usernames})


@lb_app.command("export")
def lb_export(username: str, zip_path: Path):
    """Import a Letterboxd data export zip (letterboxd.com/settings/data)."""
    from reel.sources import letterboxd

    _print(letterboxd.import_export(_lake(), username, zip_path))


# ------------------------------------------------------------------------------- transforms
@app.command("silver")
def build_silver(only: list[str] = typer.Argument(None)):
    """Rebuild silver tables from bronze."""
    from reel.transform import build_silver as run

    _print(run(_lake(), only or None))


@app.command("sql")
def sql(query: str):
    """Run DuckDB SQL over the lake. Tables are views named silver_<t> / gold_<t>."""
    lake = _lake()
    lake.register_views()
    typer.echo(lake.query(query).df().to_string() if _has_pandas() else lake.query(query))


@app.command("shell")
def shell():
    """Print a DuckDB init script so `duckdb -init <(reel shell)` opens the lake."""
    lake = _lake()
    for v in lake.register_views():
        layer, name = v.split("_", 1)
        typer.echo(f"CREATE OR REPLACE VIEW {v} AS SELECT * FROM {lake.table(layer, name)};")


# ------------------------------------------------------------------------------- ML
def _titles(lake: Lake) -> dict[int, str]:
    return dict(
        lake.query(
            f"SELECT movie_id, title || coalesce(' (' || release_year || ')', '') FROM {lake.table('silver', 'movie')}"
        ).fetchall()
    )


def _resolve(lake: Lake, movie: str) -> int:
    if movie.isdigit():
        return int(movie)
    row = lake.query(
        f"""SELECT movie_id FROM {lake.table("silver", "movie")}
            WHERE lower(title) = lower(?) OR lower(original_title) = lower(?)
            ORDER BY vote_count DESC NULLS LAST LIMIT 1""",
        params=[movie, movie],
    ).fetchone()
    if not row:
        raise typer.BadParameter(f"no movie titled {movie!r}; pass a TMDB id instead")
    return row[0]


@ml_app.command("train")
def ml_train(dim: int = 128, min_votes: int = 0):
    """Fit content embeddings (TF-IDF + LSA) and kNN neighbors."""
    from reel.ml import content

    _print(content.train(_lake(), dim=dim, min_votes=min_votes))


@ml_app.command("cluster")
def ml_cluster(min_cluster_size: int = 15, show: int = 15):
    """HDBSCAN clusters with c-TF-IDF profiles."""
    from reel.ml import cluster

    lake = _lake()
    _print(cluster.run(lake, min_cluster_size=min_cluster_size))
    rows = lake.query(
        f"SELECT cluster_id, size, top_terms, exemplars FROM {lake.table('gold', 'cluster_profile')} ORDER BY size DESC LIMIT ?",
        params=[show],
    ).fetchall()
    for cid, size, terms, ex in rows:
        typer.echo(f"\n[{cid}] n={size}  {' '.join(terms[:8])}\n     e.g. {'; '.join(ex[:5])}")


@ml_app.command("similar")
def ml_similar(movie: str, k: int = 10):
    """Nearest neighbors by content embedding. MOVIE is a title or TMDB id."""
    from reel.ml import content

    lake = _lake()
    m, titles = content.load(lake), _titles(lake)
    mid = _resolve(lake, movie)
    typer.echo(f"Similar to {titles.get(mid, mid)}:")
    for other, score in m.similar(mid, k):
        typer.echo(f"  {score:.3f}  {titles.get(other, other)}")


@ml_app.command("pairs")
def ml_pairs(movie: str, k: int = 10):
    """Double-feature suggestions with explanations."""
    from reel.ml import content
    from reel.ml.pairs import DoubleFeatureScorer

    lake = _lake()
    titles = _titles(lake)
    mid = _resolve(lake, movie)
    scorer = DoubleFeatureScorer(lake, content.load(lake))
    typer.echo(f"Double features for {titles.get(mid, mid)}:")
    for c in scorer.candidates(mid, k):
        f = c.features
        typer.echo(
            f"  {c.score:+.3f}  {titles.get(c.movie_id, c.title)}"
            f"   theme={f['theme']:.2f} contrast={int(f['contrast'])}"
            f"{' REDUNDANT' if f['redundant'] else ''}  shared: {', '.join(c.shared_keywords[:5])}"
        )


@ml_app.command("mood")
def ml_mood(query: str, k: int = 15):
    """Free-text mood/theme search (LSA query folding)."""
    from reel.ml import content

    lake = _lake()
    m, titles = content.load(lake), _titles(lake)
    hits, toks = m.search_text(query, k)
    typer.echo(f"matched tokens: {toks}")
    for mid, score in hits:
        typer.echo(f"  {score:.3f}  {titles.get(mid, mid)}")


# ------------------------------------------------------------------------------- DB
@db_app.command("migrate")
def db_migrate():
    """Apply Postgres migrations."""
    from reel.serving import db

    with db.connect(get_settings().pg_dsn) as conn:
        _print({"applied": db.migrate(conn)})


@db_app.command("publish")
def db_publish():
    """Sync silver/gold into the Postgres catalog schema."""
    from reel.serving import db, publish

    with db.connect(get_settings().pg_dsn) as conn:
        db.migrate(conn)
        _print(publish.publish(_lake(), conn))


def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    app()
