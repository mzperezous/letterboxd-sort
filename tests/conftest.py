from __future__ import annotations

import json
import random
from datetime import UTC, datetime

import httpx
import pytest

from reel.config import Settings
from reel.lake import Lake, RawRecord
from reel.storage import FsspecStore

THEMES = {
    "space": {
        "genres": ["Science Fiction", "Drama"],
        "keywords": ["space", "astronaut", "isolation", "spacecraft", "alien", "future"],
        "directors": ["Ada Orbit", "Ben Nova"],
    },
    "heist": {
        "genres": ["Crime", "Thriller"],
        "keywords": ["heist", "robbery", "double cross", "safecracker", "getaway", "money"],
        "directors": ["Cal Vault", "Dee Lock"],
    },
    "romance": {
        "genres": ["Romance", "Comedy"],
        "keywords": ["love", "wedding", "paris", "first date", "rivals to lovers", "friendship"],
        "directors": ["Eve Heart", "Fay Rose"],
    },
}


def make_movie(mid: int, theme: str, rng: random.Random, **overrides) -> dict:
    t = THEMES[theme]
    kws = rng.sample(t["keywords"], 4)
    director = rng.choice(t["directors"])
    year = rng.choice([1975, 1988, 1999, 2012, 2021])
    m = {
        "id": mid,
        "title": f"{theme.title()} Film {mid}",
        "original_title": f"{theme.title()} Film {mid}",
        "original_language": "en",
        "release_date": f"{year}-06-01",
        "runtime": rng.choice([95, 110, 125]),
        "overview": f"A {theme} story.",
        "tagline": "",
        "status": "Released",
        "vote_average": round(rng.uniform(5.5, 8.5), 1),
        "vote_count": rng.randint(50, 5000),
        "popularity": rng.uniform(1, 100),
        "budget": 0,
        "revenue": 0,
        "poster_path": f"/p{mid}.jpg",
        "imdb_id": f"tt{mid:07d}",
        "belongs_to_collection": None,
        "genres": [{"id": i, "name": g} for i, g in enumerate(t["genres"])],
        "production_countries": [{"iso_3166_1": "US", "name": "United States of America"}],
        "keywords": {"keywords": [{"id": hash(k) % 10000, "name": k} for k in kws]},
        "external_ids": {"imdb_id": f"tt{mid:07d}", "wikidata_id": f"Q{mid}"},
        "credits": {
            "cast": [
                {
                    "id": 9000 + mid % 7,
                    "name": f"Actor {theme} {mid % 3}",
                    "character": "x",
                    "order": 0,
                    "gender": 1,
                }
            ],
            "crew": [
                {
                    "id": 500 + hash(director) % 50,
                    "name": director,
                    "job": "Director",
                    "department": "Directing",
                }
            ],
        },
        "release_dates": {"results": []},
    }
    m.update(overrides)
    return m


@pytest.fixture
def lake(tmp_path) -> Lake:
    return Lake(
        FsspecStore((tmp_path / "lake").as_uri()), Settings(lake_url=(tmp_path / "lake").as_uri())
    )


@pytest.fixture
def movies() -> dict[int, dict]:
    rng = random.Random(42)
    out = {}
    mid = 100
    for theme in THEMES:
        for _ in range(25):
            out[mid] = make_movie(mid, theme, rng)
            mid += 1
    # A sequel pair that should be flagged redundant for double features.
    out[100]["belongs_to_collection"] = {"id": 77, "name": "Space Saga"}
    out[101]["belongs_to_collection"] = {"id": 77, "name": "Space Saga"}
    return out


@pytest.fixture
def seeded_lake(lake, movies) -> Lake:
    from reel.sources.tmdb import trim_movie
    from reel.transform import build_silver

    t0 = datetime(2026, 11, 1, tzinfo=UTC)
    lake.write_bronze(
        "tmdb",
        "movie",
        [RawRecord(str(k), trim_movie(v), fetched_at=t0) for k, v in movies.items()],
    )
    build_silver(lake)
    return lake


class FakeTmdb:
    """httpx MockTransport handler emulating the TMDB endpoints we use."""

    def __init__(self, movies: dict[int, dict]):
        self.movies = movies
        self.changed: list[int] = []
        self.calls: list[str] = []
        self.fail_first: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if path in self.fail_first:
            self.fail_first.discard(path)
            return httpx.Response(429, headers={"retry-after": "0"})
        if path.startswith("/3/movie/changes"):
            page = int(request.url.params.get("page", 1))
            per = 2
            chunk = self.changed[(page - 1) * per : page * per]
            total = max(1, -(-len(self.changed) // per))
            return httpx.Response(
                200,
                json={
                    "results": [{"id": i, "adult": False} for i in chunk],
                    "page": page,
                    "total_pages": total,
                },
            )
        if path.startswith("/3/movie/"):
            mid = int(path.rsplit("/", 1)[-1])
            if mid not in self.movies:
                return httpx.Response(404, json={"status_code": 34})
            return httpx.Response(200, content=json.dumps(self.movies[mid]))
        if path.startswith("/3/discover/movie"):
            page = int(request.url.params.get("page", 1))
            ids = sorted(self.movies)
            chunk = ids[(page - 1) * 20 : page * 20]
            return httpx.Response(200, json={"results": [{"id": i} for i in chunk]})
        return httpx.Response(404)


@pytest.fixture
def fake_tmdb(movies):
    return FakeTmdb(movies)


@pytest.fixture
def tmdb_client(fake_tmdb):
    from reel.sources.tmdb import TmdbClient

    s = Settings(tmdb_requests_per_second=1000, tmdb_concurrency=4)
    return TmdbClient(s, transport=httpx.MockTransport(fake_tmdb))
