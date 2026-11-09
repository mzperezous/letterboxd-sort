import asyncio
import io
import zipfile
from datetime import date

from reel.lake import Lake
from reel.sources import letterboxd, movielens, tmdb
from reel.transform import build_silver


def test_trim_movie_drops_volatile_fields(movies):
    m = movies[100] | {
        "credits": {
            "cast": [{"id": i, "name": f"a{i}", "order": i} for i in range(40)],
            "crew": [
                {"id": 1, "name": "D", "job": "Director"},
                {"id": 2, "name": "G", "job": "Gaffer"},
            ],
        }
    }
    t = tmdb.trim_movie(m)
    assert "popularity" not in t
    assert len(t["credits"]["cast"]) == tmdb.MAX_CAST
    assert [c["job"] for c in t["credits"]["crew"]] == ["Director"]


def test_ingest_movies_and_change_detection(lake: Lake, tmdb_client, fake_tmdb, movies):
    async def run(ids, **kw):
        return await tmdb.ingest_movies(lake, tmdb_client, ids, **kw)

    first = asyncio.run(run([100, 101, 102, 99999]))
    assert first["written"] == 3 and first["not_found"] == 1  # unknown 404 -> no tombstone

    again = asyncio.run(run([100, 101, 102]))
    assert again["written"] == 0 and again["unchanged"] == 3

    movies[101]["title"] = "Renamed"
    del movies[102]  # deleted upstream
    third = asyncio.run(run([100, 101, 102]))
    assert third["written"] == 2  # new version of 101 + tombstone for 102

    build_silver(lake)
    hist = lake.query(
        f"SELECT movie_id, version, is_deleted FROM {lake.table('silver', 'tmdb_movie_history')} WHERE is_current ORDER BY 1"
    ).fetchall()
    assert hist == [(100, 1, False), (101, 2, False), (102, 2, True)]
    titles = lake.query(
        f"SELECT movie_id, title FROM {lake.table('silver', 'movie')} ORDER BY 1"
    ).fetchall()
    assert titles == [(100, "Space Film 100"), (101, "Renamed")]


def test_only_missing_skips_fetched(lake, tmdb_client, fake_tmdb):
    asyncio.run(tmdb.ingest_movies(lake, tmdb_client, [100]))
    fake_tmdb.calls.clear()
    asyncio.run(tmdb.ingest_movies(lake, tmdb_client, [100, 101], only_missing=True))
    assert fake_tmdb.calls == ["/3/movie/101"]


def test_retry_on_429(lake, tmdb_client, fake_tmdb):
    fake_tmdb.fail_first.add("/3/movie/100")
    stats = asyncio.run(tmdb.ingest_movies(lake, tmdb_client, [100]))
    assert stats["written"] == 1
    assert fake_tmdb.calls.count("/3/movie/100") == 2


def test_changes_feed_refetches_only_catalog(lake, tmdb_client, fake_tmdb, movies):
    asyncio.run(tmdb.ingest_movies(lake, tmdb_client, [100, 101, 102]))
    movies[101]["overview"] = "changed"
    fake_tmdb.changed = [101, 102, 150, 151, 152]  # 150+ not in our catalog
    stats = asyncio.run(
        tmdb.ingest_changes(lake, tmdb_client, date(2026, 11, 1), date(2026, 11, 2))
    )
    assert stats["changed_upstream"] == 5
    assert stats["in_catalog"] == 2
    assert stats["written"] == 1  # 102 changed "upstream" but content identical


def test_movielens_ingest_and_silver(lake, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "ml-latest-small/movies.csv",
            "movieId,title,genres\n1,Toy Story (1995),Animation|Comedy\n2,Heat (1995),Crime\n",
        )
        zf.writestr(
            "ml-latest-small/links.csv", "movieId,imdbId,tmdbId\n1,114709,862\n2,113277,949\n"
        )
        zf.writestr(
            "ml-latest-small/ratings.csv",
            "userId,movieId,rating,timestamp\n1,1,4.0,964982703\n1,2,5.0,964982703\n",
        )
        zf.writestr(
            "ml-latest-small/tags.csv", "userId,movieId,tag,timestamp\n1,2,Slow Burn ,964982703\n"
        )
    zp = tmp_path / "ml.zip"
    zp.write_bytes(buf.getvalue())
    from reel.config import Settings

    out = movielens.ingest(lake, Settings(), "ml-latest-small", zip_path=zp)
    assert out["files"] == ["links.csv", "movies.csv", "ratings.csv", "tags.csv"]
    build_silver(lake, ["ml_movie", "ml_rating", "ml_tag"])
    rows = lake.query(
        f"SELECT ml_movie_id, ml_year, movie_id, imdb_id, ml_genres FROM {lake.table('silver', 'ml_movie')} ORDER BY 1"
    ).fetchall()
    assert rows[0] == (1, 1995, 862, "tt0114709", ["Animation", "Comedy"])
    assert lake.query(f"SELECT tag FROM {lake.table('silver', 'ml_tag')}").fetchone() == (
        "slow burn",
    )


RSS = """<?xml version='1.0' encoding='utf-8'?>
<rss version="2.0" xmlns:letterboxd="https://letterboxd.com" xmlns:tmdb="https://themoviedb.org"><channel>
<item><title>Heat, 1995 - ★★★★★</title><link>https://letterboxd.com/me/film/heat/</link>
<guid isPermaLink="false">letterboxd-review-1</guid><pubDate>Fri, 6 Nov 2026 06:39:36 +1300</pubDate>
<letterboxd:watchedDate>2026-11-05</letterboxd:watchedDate><letterboxd:rewatch>Yes</letterboxd:rewatch>
<letterboxd:filmTitle>Heat</letterboxd:filmTitle><letterboxd:filmYear>1995</letterboxd:filmYear>
<letterboxd:memberRating>5.0</letterboxd:memberRating><letterboxd:memberLike>Yes</letterboxd:memberLike>
<tmdb:movieId>949</tmdb:movieId>
<description><![CDATA[ <p><img src="x.jpg"/></p> <p>Melancholy <b>and</b> tense.</p> ]]></description></item>
<item><title>A list</title><link>https://letterboxd.com/me/list/x/</link><guid>list-1</guid></item>
</channel></rss>"""


def test_parse_rss():
    items = letterboxd.parse_rss(RSS, "me")
    assert len(items) == 1
    i = items[0]
    assert (i["tmdb_id"], i["rating"], i["rewatch"], i["liked"]) == (949, 5.0, True, True)
    assert i["review_text"] == "Melancholy and tense."


def test_letterboxd_export_reimport_is_cdc(lake, tmp_path):
    def export(rating):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "ratings.csv",
                f"Date,Name,Year,Letterboxd URI,Rating\n2026-01-01,Heat,1995,https://boxd.it/abc,{rating}\n",
            )
            zf.writestr(
                "watched.csv",
                "Date,Name,Year,Letterboxd URI\n2026-01-01,Heat,1995,https://boxd.it/abc\n",
            )
        p = tmp_path / f"e{rating}.zip"
        p.write_bytes(buf.getvalue())
        return p

    first = letterboxd.import_export(lake, "Me", export(4))
    assert first["rating"]["written"] == 1
    again = letterboxd.import_export(lake, "me", export(4))
    assert again["rating"]["written"] == 0 and again["watched"]["written"] == 0
    letterboxd.import_export(lake, "me", export(4.5))
    build_silver(lake, ["lb_rating"])
    assert lake.query(f"SELECT rating FROM {lake.table('silver', 'lb_rating')}").fetchall() == [
        (4.5,)
    ]


def test_username_validation():
    import pytest

    with pytest.raises(ValueError):
        letterboxd.check_username("../etc/passwd")
