"""Letterboxd: a user's data export (complete history) and public RSS feed (recent diary +
review text, and crucially a TMDB id per film).

Each row becomes a bronze record keyed by user + film, so re-importing an export later yields a
change log of that user's ratings/watches (CDC on user taste for free).

Be polite: Letterboxd's terms don't permit bulk scraping, and most pages sit behind Cloudflare.
RSS and the user's own export are the sanctioned-feeling paths; keep request rates very low.
"""

from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from reel.config import Settings
from reel.lake import Lake, RawRecord
from reel.sources.http import USER_AGENT

SOURCE = "letterboxd"
NS = {"letterboxd": "https://letterboxd.com", "tmdb": "https://themoviedb.org"}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{2,30}$")

EXPORT_FILES = {
    "watched.csv": "watched",
    "ratings.csv": "rating",
    "diary.csv": "diary",
    "watchlist.csv": "watchlist",
    "reviews.csv": "review",
    "likes/films.csv": "like",
}


def check_username(username: str) -> str:
    if not USERNAME_RE.match(username):
        raise ValueError(f"invalid letterboxd username: {username!r}")
    return username.lower()


def _row_key(username: str, entity: str, row: dict) -> str:
    uri = row.get("Letterboxd URI") or row.get("Name", "")
    # Diary/review rows can repeat a film (rewatches); include the watched/entry date.
    if entity in ("diary", "review"):
        return f"{username}|{uri}|{row.get('Watched Date') or row.get('Date')}"
    return f"{username}|{uri}"


def import_export(lake: Lake, username: str, zip_path: Path) -> dict:
    username = check_username(username)
    raw = zip_path.read_bytes()
    stamp = datetime.now(UTC)
    lake.put_file(SOURCE, f"{username}/export-{stamp:%Y%m%dT%H%M%S}.zip", raw)
    out: dict[str, dict] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        for fname, entity in EXPORT_FILES.items():
            if fname not in names:
                continue
            text = zf.read(names[fname]).decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(text)))
            records = [
                RawRecord(
                    key=_row_key(username, entity, r),
                    payload={"username": username, **r},
                    fetched_at=stamp,
                )
                for r in rows
            ]
            out[entity] = lake.write_bronze(SOURCE, f"export_{entity}", records)
    return out


def parse_rss(xml_text: str, username: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):

        def t(tag: str, item=item) -> str | None:
            el = item.find(tag, NS)
            return el.text.strip() if el is not None and el.text else None

        title = t("letterboxd:filmTitle")
        if not title:
            continue  # list entries etc.
        desc = t("description") or ""
        review_html = re.sub(r"<p><img[^>]*/></p>", "", desc).strip()
        review = re.sub(r"<[^>]+>", " ", review_html)
        review = re.sub(r"\s+", " ", review).strip()
        pub = t("pubDate")
        items.append(
            {
                "username": username,
                "guid": t("guid"),
                "link": t("link"),
                "film_title": title,
                "film_year": int(t("letterboxd:filmYear")) if t("letterboxd:filmYear") else None,
                "tmdb_id": int(t("tmdb:movieId")) if t("tmdb:movieId") else None,
                "watched_date": t("letterboxd:watchedDate"),
                "rewatch": t("letterboxd:rewatch") == "Yes",
                "rating": float(t("letterboxd:memberRating"))
                if t("letterboxd:memberRating")
                else None,
                "liked": t("letterboxd:memberLike") == "Yes",
                "published_at": parsedate_to_datetime(pub).isoformat() if pub else None,
                "review_text": None if review.startswith("Watched on ") else review or None,
            }
        )
    return items


def fetch_rss(settings: Settings, username: str) -> str:
    username = check_username(username)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as c:
        resp = c.get(f"{settings.letterboxd_base_url}/{username}/rss/", follow_redirects=True)
        resp.raise_for_status()
        return resp.text


def ingest_rss(lake: Lake, settings: Settings, username: str) -> dict:
    username = check_username(username)
    items = parse_rss(fetch_rss(settings, username), username)
    records = [RawRecord(key=i["guid"] or i["link"], payload=i) for i in items]
    return lake.write_bronze(SOURCE, "rss_entry", records)
