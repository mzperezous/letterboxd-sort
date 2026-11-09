"""Typed settings. Everything environment-specific lives here and is read from REEL_* env vars
(or a .env file), so dev -> prod is a config change, not a code change."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REEL_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    # --- Object storage (the lake). file://, s3://, or memory:// ---
    lake_url: str = Field(default_factory=lambda: (Path.cwd() / "data" / "lake").as_uri())
    # S3-compatible settings (MinIO locally, S3/R2/GCS-interop in prod). Unused for file://.
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_region: str = "us-east-1"

    # --- Serving database ---
    pg_dsn: str = "postgresql://reel:reel@localhost:5433/reel"

    # --- TMDB ---
    # Optional: in Claude Code cloud sessions the network proxy injects the bearer token, so
    # leaving this unset is fine there. Locally, set REEL_TMDB_TOKEN to your Read Access Token.
    tmdb_token: SecretStr | None = None
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_export_base_url: str = "https://files.tmdb.org/p/exports"
    tmdb_requests_per_second: float = 3.5  # TMDB limit is 40 req / 10s; keep headroom
    tmdb_concurrency: int = 4

    # --- MovieLens ---
    movielens_base_url: str = "https://files.grouplens.org/datasets/movielens"
    movielens_variant: str = "ml-latest-small"  # ml-latest-small | ml-25m | ml-32m
    # Path to an extra CA bundle if the host's chain is not in your trust store.
    movielens_ca_bundle: str | None = None

    # --- Letterboxd ---
    letterboxd_base_url: str = "https://letterboxd.com"
    letterboxd_requests_per_second: float = 1.0
    # Comma-separated usernames whose RSS the scheduled pipeline ingests.
    letterboxd_users: str = ""

    # --- Seed catalog sizing ---
    seed_popular: int = 2000
    seed_top_voted: int = 10000

    @property
    def is_s3(self) -> bool:
        return self.lake_url.startswith("s3://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
