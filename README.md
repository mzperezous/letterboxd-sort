# reel (working name)

A film data platform for finding **double features** and **mood-based** picks. It ingests TMDB,
MovieLens and Letterboxd into an S3-style lake with change capture, builds normalized tables
with DuckDB, trains deterministic open-source models (TF-IDF/LSA, HDBSCAN, Bradley-Terry), and
serves them from Postgres + pgvector through a FastAPI service.

> This branch is a greenfield rewrite. The original Beli-style ranking prototype lives on
> `main`; its comparison flow comes back as a feature of the new app.

See [docs/architecture.md](docs/architecture.md) for the design: layers, CDC strategy, dev → prod
path and the modeling roadmap.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```sh
make install                 # deps (incl. dagster, dev tools)
cp .env.example .env         # set REEL_TMDB_TOKEN (TMDB "API Read Access Token")
make up                      # Postgres + pgvector on :5433 (PROFILE=s3 adds MinIO)

make ingest                  # TMDB id export + MovieLens + ~12k film seed (~1 h at TMDB's
                             # 40 req/10s; REEL_SEED_TOP_VOTED=1000 for a 5-min loop)
make build                   # silver tables → embeddings → clusters → Postgres
make api                     # http://localhost:8000/docs
```

Or drive everything from the Dagster UI: `make dagster` → http://localhost:3000, materialize the
`bootstrap` job. The `daily_refresh` schedule then keeps the catalog current.

## Playing with it

```sh
uv run reel ml similar "Heat"
uv run reel ml pairs "Alien"                    # double features, with reasons
uv run reel ml mood "lonely melancholy space"   # mood/theme search, no LLM
uv run reel ml cluster --show 20                # discovered clusters + their profiles
uv run reel sql "SELECT title, vote_count FROM silver_movie ORDER BY vote_count DESC LIMIT 10"
uv run reel letterboxd rss <username>           # recent diary + review text (has TMDB ids)
uv run reel letterboxd export <username> ~/Downloads/letterboxd-<user>.zip
uv run reel tmdb changes                        # CDC: refresh films TMDB changed yesterday
```

API highlights (`make api`, then `/docs`): `GET /movies/search?q=`, `/movies/{id}/similar`
(pgvector), `/movies/{id}/double-features`, `/mood?q=`, `/clusters`,
`POST /users/{name}/comparisons`, `GET /users/{name}/recommendations`.

## Layout

```
src/reel/
  config.py          REEL_* settings (dev/prod is config, not code)
  storage.py         ObjectStore abstraction: file:// | s3:// | memory://
  lake.py            bronze writer w/ content-hash CDC, DuckDB warehouse over the store
  transform.py       silver tables (SCD2 + normalized), dependency-ordered registry
  sources/           tmdb.py, movielens.py, letterboxd.py, http.py (rate limit + retry)
  ml/                content.py (TF-IDF→LSA, kNN, query folding), cluster.py (HDBSCAN,
                     c-TF-IDF), pairs.py (double features), taste.py (Bradley-Terry)
  serving/           Postgres migrations, lake→Postgres sync
  api/app.py         FastAPI
  pipelines/         Dagster assets, jobs, schedule
  cli.py             `reel` command
infra/docker-compose.yml   Postgres+pgvector, MinIO
tests/                     fast, offline (fake TMDB transport); Postgres tests if REEL_TEST_PG_DSN
```

## Tests

```sh
make test                                                     # offline
REEL_TEST_PG_DSN=postgresql://reel:reel@localhost:5433/reel_test make test   # + Postgres
make lint
```

## Data sources & terms

- **TMDB**: free API for non-commercial use; attribution required ("This product uses the TMDB
  API but is not endorsed or certified by TMDB").
- **MovieLens** (GroupLens): research/non-commercial license; cite Harper & Konstan (2015).
- **Letterboxd**: no public API without approval, and the terms prohibit scraping. This project
  only uses a user's own data export and public RSS, at very low rates.
