"""TMDB source.

Three inputs, which together give full change capture without TMDB ever pushing to us:

1. Daily ID export (files.tmdb.org, no auth): every movie id + popularity. Diffing consecutive
   days tells us about new and removed ids. Stored in bronze as tmdb/id_export, keyed by id.
2. Changes feed (/movie/changes): ids whose data changed in a window (max 14 days back).
   This is our CDC stream: changed ids that are in our catalog get re-fetched.
3. Movie details (/movie/{id} + append_to_response): the actual payload, written to bronze
   tmdb/movie with content hashing, so only real changes produce new versions.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, date, datetime, timedelta

import httpx

from reel.config import Settings
from reel.lake import Lake, RawRecord
from reel.sources.http import USER_AGENT, RateLimiter, get_with_retry

log = logging.getLogger(__name__)

SOURCE = "tmdb"
DETAIL_APPEND = "keywords,credits,external_ids,release_dates"
# Crew jobs worth keeping; full crew lists are huge and mostly irrelevant for taste.
KEEP_CREW_JOBS = {
    "Director",
    "Screenplay",
    "Writer",
    "Director of Photography",
    "Original Music Composer",
    "Editor",
    "Producer",
    "Novel",
}
MAX_CAST = 15


class TmdbClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if settings.tmdb_token:
            headers["Authorization"] = f"Bearer {settings.tmdb_token.get_secret_value()}"
        self.client = httpx.AsyncClient(
            headers=headers, timeout=httpx.Timeout(30.0), transport=transport
        )
        # burst=1: never more than rate*window (+1) requests in any window.
        self.limiter = RateLimiter(settings.tmdb_requests_per_second, burst=1)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.client.aclose()

    # ------------------------------------------------------------------ exports
    async def id_export(self, day: date) -> bytes:
        """Raw gzipped JSON-lines export for `day`."""
        url = f"{self.s.tmdb_export_base_url}/movie_ids_{day:%m_%d_%Y}.json.gz"
        resp = await get_with_retry(self.client, url)
        resp.raise_for_status()
        return resp.content

    async def latest_id_export(self, max_days_back: int = 3) -> tuple[date, bytes]:
        """Exports appear ~08:00 UTC; walk back until one exists."""
        today = datetime.now(UTC).date()
        for back in range(max_days_back + 1):
            day = today - timedelta(days=back)
            try:
                return day, await self.id_export(day)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in (403, 404):
                    raise
        raise RuntimeError("no TMDB id export found in the last few days")

    # ------------------------------------------------------------------ api
    async def movie(self, movie_id: int) -> tuple[int, dict | None]:
        resp = await get_with_retry(
            self.client,
            f"{self.s.tmdb_base_url}/movie/{movie_id}",
            limiter=self.limiter,
            params={"append_to_response": DETAIL_APPEND},
        )
        if resp.status_code == 404:
            return 404, None
        resp.raise_for_status()
        return 200, trim_movie(resp.json())

    async def changes(self, start: date, end: date) -> list[int]:
        ids: list[int] = []
        page, total = 1, 1
        while page <= total:
            resp = await get_with_retry(
                self.client,
                f"{self.s.tmdb_base_url}/movie/changes",
                limiter=self.limiter,
                params={"start_date": start.isoformat(), "end_date": end.isoformat(), "page": page},
            )
            resp.raise_for_status()
            body = resp.json()
            total = min(body.get("total_pages", 1), 500)
            ids += [r["id"] for r in body.get("results", []) if not r.get("adult")]
            page += 1
        return ids

    async def discover_ids(self, pages: int, sort_by: str = "vote_count.desc") -> list[int]:
        """Ids from /discover/movie (20 per page, max 500 pages). vote_count.desc ~ the canon."""
        ids: list[int] = []
        for page in range(1, min(pages, 500) + 1):
            resp = await get_with_retry(
                self.client,
                f"{self.s.tmdb_base_url}/discover/movie",
                limiter=self.limiter,
                params={"sort_by": sort_by, "include_adult": "false", "page": page},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                break
            ids += [r["id"] for r in results]
        return ids

    async def search(self, title: str, year: int | None = None) -> dict:
        params: dict = {"query": title, "include_adult": "false"}
        if year:
            params["year"] = year
        resp = await get_with_retry(
            self.client, f"{self.s.tmdb_base_url}/search/movie", limiter=self.limiter, params=params
        )
        resp.raise_for_status()
        return resp.json()

    async def movies(self, ids: Iterable[int]) -> AsyncIterator[RawRecord]:
        """Fetch many movies concurrently, yielding records as they complete."""
        sem = asyncio.Semaphore(self.s.tmdb_concurrency)

        async def one(mid: int) -> RawRecord:
            async with sem:
                status, payload = await self.movie(mid)
                return RawRecord(key=str(mid), payload=payload, status=status)

        tasks = [asyncio.create_task(one(i)) for i in ids]
        try:
            for fut in asyncio.as_completed(tasks):
                yield await fut
        finally:
            for t in tasks:
                t.cancel()


def trim_movie(m: dict) -> dict:
    """Drop volatile/bulky fields before hashing so versions reflect meaningful change.

    `popularity` moves every day; keeping it would create a new version per movie per day. It is
    captured separately (and cheaply) from the daily id export instead.
    """
    m = dict(m)
    m.pop("popularity", None)
    credits = m.get("credits") or {}
    m["credits"] = {
        "cast": [
            {k: c.get(k) for k in ("id", "name", "character", "order", "gender")}
            for c in sorted(credits.get("cast", []), key=lambda c: c.get("order", 999))[:MAX_CAST]
        ],
        "crew": [
            {k: c.get(k) for k in ("id", "name", "job", "department")}
            for c in credits.get("crew", [])
            if c.get("job") in KEEP_CREW_JOBS
        ],
    }
    rd = (m.get("release_dates") or {}).get("results", [])
    m["release_dates"] = {"results": [r for r in rd if r.get("iso_3166_1") in {"US", "GB", "FR"}]}
    return m


# ---------------------------------------------------------------------- pipeline operations


def export_key(day: date) -> str:
    return f"id_exports/movie_ids_{day.isoformat()}.json.gz"


def parse_export(raw: bytes) -> list[dict]:
    lines = gzip.decompress(raw).decode().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def prune_exports(lake: Lake, keep: int = 7) -> list[str]:
    """Raw exports are ~28 MB/day; history already lives in bronze/tmdb/id_export, so only the
    last `keep` raw files are retained (for popularity ranking)."""
    keys = sorted(
        k for k in lake.store.list(f"files/{SOURCE}/id_exports") if k.endswith(".json.gz")
    )
    stale = keys[:-keep] if keep else keys
    for k in stale:
        lake.store.delete(k)
    return stale


async def ingest_id_export(
    lake: Lake, client: TmdbClient, day: date | None = None, keep_raw: int = 7
) -> dict:
    """Store the raw export file (used for popularity / seed ranking) and append id-level
    changes (new, retitled, deleted ids) to bronze tmdb/id_export."""
    if day is None:
        day, raw = await client.latest_id_export()
    else:
        raw = await client.id_export(day)
    lake.put_file(SOURCE, export_key(day), raw)
    rows = parse_export(raw)
    fetched_at = datetime(day.year, day.month, day.day, 8, tzinfo=UTC)
    current = {str(r["id"]) for r in rows}
    latest = lake.latest_hashes(SOURCE, "id_export")

    def records():
        for r in rows:
            # Popularity is deliberately excluded: it changes daily and lives in the raw file.
            payload = {
                "id": r["id"],
                "original_title": r.get("original_title"),
                "adult": r.get("adult", False),
                "video": r.get("video", False),
            }
            yield RawRecord(key=str(r["id"]), payload=payload, fetched_at=fetched_at)
        # Ids that disappeared from the export were deleted upstream.
        for key, h in latest.items():
            if key not in current and h != "tombstone":
                yield RawRecord(key=key, payload=None, status=404, fetched_at=fetched_at)

    stats = lake.write_bronze(SOURCE, "id_export", records(), latest=latest)
    pruned = prune_exports(lake, keep_raw)
    return {"export_date": day.isoformat(), "rows": len(rows), "pruned_raw": len(pruned), **stats}


async def discover_ids_cached(lake: Lake, client: TmdbClient, n: int) -> list[int]:
    """Top-n ids by vote count, cached per day in the lake (500 pages = 2.5 min at 3.5 req/s)."""
    key = f"discover/vote_count_{datetime.now(UTC).date().isoformat()}.json"
    full_key = f"files/{SOURCE}/{key}"
    if lake.store.exists(full_key):
        cached = json.loads(lake.store.get(full_key))
        if len(cached) >= n:
            return cached[:n]
    ids = await client.discover_ids(pages=(n + 19) // 20)
    lake.put_file(SOURCE, key, json.dumps(ids).encode())
    return ids[:n]


def latest_export_uri(lake: Lake) -> str:
    keys = [k for k in lake.store.list(f"files/{SOURCE}/id_exports") if k.endswith(".json.gz")]
    if not keys:
        raise LookupError("no TMDB id export ingested yet; run `reel tmdb export` first")
    return lake.store.uri(sorted(keys)[-1])


def select_seed_ids(lake: Lake, limit: int, extra_ids: Iterable[int] = ()) -> list[int]:
    """Top-N non-adult, non-video ids by popularity from the latest export, plus any extra ids
    (e.g. films from MovieLens or a user's Letterboxd history)."""
    rows = lake.query(
        f"""
        SELECT id
        FROM read_json('{latest_export_uri(lake)}', format = 'newline_delimited',
                       columns = {{id: 'BIGINT', adult: 'BOOLEAN', video: 'BOOLEAN',
                                   popularity: 'DOUBLE', original_title: 'VARCHAR'}})
        WHERE NOT coalesce(adult, false) AND NOT coalesce(video, false)
        ORDER BY popularity DESC, id
        LIMIT ?
        """,
        params=[limit],
    ).fetchall()
    ids = [r[0] for r in rows]
    seen = set(ids)
    ids += [i for i in extra_ids if i not in seen]
    return ids


async def ingest_movies(
    lake: Lake, client: TmdbClient, ids: Iterable[int], *, only_missing: bool = False
) -> dict:
    ids = list(dict.fromkeys(int(i) for i in ids))
    latest = lake.latest_hashes(SOURCE, "movie")
    if only_missing:
        ids = [i for i in ids if str(i) not in latest]
    log.info("fetching %d tmdb movies", len(ids))
    # Bridge async fetching to the synchronous, chunked bronze writer.
    records: list[RawRecord] = []
    totals = {"requested": len(ids), "seen": 0, "written": 0, "unchanged": 0, "not_found": 0}
    async for rec in client.movies(ids):
        if rec.status == 404:
            totals["not_found"] += 1
        records.append(rec)
        if len(records) >= 500:
            for k, v in lake.write_bronze(SOURCE, "movie", records, latest=latest).items():
                totals[k] += v
            records = []
    for k, v in lake.write_bronze(SOURCE, "movie", records, latest=latest).items():
        totals[k] += v
    return totals


def catalog_ids(lake: Lake) -> set[int]:
    if not lake.bronze_exists(SOURCE, "movie"):
        return set()
    rows = lake.query(f"SELECT DISTINCT CAST(key AS BIGINT) FROM {lake.bronze(SOURCE, 'movie')}")
    return {r[0] for r in rows.fetchall()}


async def ingest_changes(lake: Lake, client: TmdbClient, start: date, end: date) -> dict:
    """CDC: re-fetch catalog movies that TMDB reports as changed in [start, end]."""
    changed = set(await client.changes(start, end))
    targets = sorted(changed & catalog_ids(lake))
    stats = await ingest_movies(lake, client, targets)
    return {"changed_upstream": len(changed), "in_catalog": len(targets), **stats}
