"""MovieLens (GroupLens) static dumps.

Bronze here is *file-level*: the zip is immutable and versioned by variant, so we store the zip
and its extracted CSVs under files/movielens/{variant}/ and let silver read the CSVs directly.

Variants worth knowing:
  ml-latest-small  ~100k ratings, 9.7k movies. Fast dev loop.
  ml-25m           25M ratings + the Tag Genome (1,128 tags x 13.8k movies, relevance 0..1).
                   The genome is the best free, dense "mood/vibe" signal there is.
  ml-32m           32M ratings, newer, no genome.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx

from reel.config import Settings
from reel.lake import Lake
from reel.sources.http import USER_AGENT

SOURCE = "movielens"
WANTED = {
    "movies.csv",
    "ratings.csv",
    "tags.csv",
    "links.csv",
    "genome-scores.csv",
    "genome-tags.csv",
}


def download(settings: Settings, variant: str) -> bytes:
    url = f"{settings.movielens_base_url}/{variant}.zip"
    verify: str | bool = settings.movielens_ca_bundle or True
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=600, verify=verify) as c:
        resp = c.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.content


def ingest(
    lake: Lake, settings: Settings, variant: str | None = None, zip_path: Path | None = None
):
    """Download (or read a local copy of) a MovieLens zip and land its CSVs in the lake."""
    variant = variant or settings.movielens_variant
    raw = zip_path.read_bytes() if zip_path else download(settings, variant)
    lake.put_file(SOURCE, f"{variant}.zip", raw)
    landed = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            if base in WANTED:
                lake.put_file(SOURCE, f"{variant}/{base}", zf.read(name))
                landed.append(base)
    return {"variant": variant, "files": sorted(landed), "bytes": len(raw)}


def csv_uri(lake: Lake, variant: str, name: str) -> str | None:
    key = f"files/{SOURCE}/{variant}/{name}"
    return lake.store.uri(key) if lake.store.exists(key) else None


def ingested_variants(lake: Lake) -> list[str]:
    variants = set()
    for key in lake.store.list(f"files/{SOURCE}"):
        parts = key.split("/")
        if len(parts) == 4 and parts[3] == "movies.csv":
            variants.add(parts[2])
    return sorted(variants)
