# CONTEXT

Working context for this repo: what it is, how it's put together, conventions, gotchas, and
links to the primary docs for every source and tool. Read this before changing anything.
Design rationale lives in [docs/architecture.md](docs/architecture.md); this file is the
operational companion.

## What this is

A personal film data platform (working name **reel**; the repo will be renamed). Goals:

1. A film catalog that stays current via change capture, not full reloads.
2. Reusable pipeline/ML infrastructure: dev-first, with production as a config change.
3. Products: **double-feature finder**, **mood search**, later personalized from
   Letterboxd profiles and in-app Beli-style rankings.

Bias: **deterministic, cheap, open-source ML first** (TF-IDF, SVD, HDBSCAN, logistic
regression). LLMs only where they clearly earn it (cluster naming, blurbs), cached.

Branch `claude/brave-curie-s1ep0b` is the greenfield rewrite; `main` still holds the original
ranking prototype and will be overwritten later.

## Map

| Path | Role |
|---|---|
| `src/reel/config.py` | All settings, `REEL_*` env vars / `.env` (pydantic-settings) |
| `src/reel/storage.py` | `ObjectStore` protocol + `FsspecStore` (file, s3, memory) |
| `src/reel/lake.py` | Bronze writer (content-hash CDC, tombstones), `scd2_sql()`, silver/gold table IO, DuckDB |
| `src/reel/transform.py` | Silver tables as SQL, registered with `@silver(...)` |
| `src/reel/sources/` | `tmdb.py`, `movielens.py`, `letterboxd.py`, `http.py` (rate limit, retries) |
| `src/reel/ml/` | `content.py` (TF-IDF→LSA, kNN, query folding), `cluster.py`, `pairs.py`, `taste.py` |
| `src/reel/serving/` | Postgres migrations (`migrations/NNN_*.sql`), `sync_table`, `publish` |
| `src/reel/api/app.py` | FastAPI app |
| `src/reel/pipelines/definitions.py` | Dagster assets/jobs/schedule (call the same functions as the CLI) |
| `src/reel/cli.py` | `reel` CLI (typer) |
| `tests/` | Offline tests with a fake TMDB transport; Postgres tests gated on `REEL_TEST_PG_DSN` |

## Lake layout

```
bronze/{source}/{entity}/ingest_date=YYYY-MM-DD/{run_id}-{n}.parquet   append-only change log
files/{source}/...                                                     raw dumps/exports
silver/{table}/data.parquet                                            rebuilt from bronze
gold/{table}/data.parquet                                              embeddings, kNN, clusters
models/{name}/model.joblib                                             fitted artifacts
```

Bronze columns: `source, entity, key, fetched_at, run_id, status, content_hash, payload`
(payload is canonical JSON; tombstones have `status=404`, `payload=NULL`,
`content_hash='tombstone'`).

## Invariants (don't break these)

- **Bronze is append-only.** Never rewrite or delete bronze files (raw `files/` may be pruned).
- **Silver/gold are pure functions of bronze**, rebuilt in full. No incremental merges yet.
- **Strip volatile fields before hashing** (e.g. `tmdb.trim_movie` drops `popularity`) or
  every fetch becomes a new version.
- **Postgres `catalog.*` is a disposable projection**; `app.*` is source of truth for user
  actions (append-only `app.event`).
- **One code path:** Dagster assets and CLI commands call the same library functions.
- **Deterministic ML:** fixed `random_state`, stable sorts (`kind="stable"`), tie-break by id.

## Commands

```sh
make install / up / ingest / build / api / dagster / test / lint / fmt
uv run reel --help
uv run reel tmdb export | seed | fetch ID... | changes
uv run reel movielens [--variant ml-25m] [--zip path]
uv run reel letterboxd rss USER | export USER ZIP
uv run reel silver [TABLE...]
uv run reel ml train | cluster | similar TITLE | pairs TITLE | mood "TEXT"
uv run reel db migrate | publish
uv run reel sql "SELECT ... FROM silver_movie"     # views: silver_<t>, gold_<t>
REEL_TEST_PG_DSN=postgresql://reel:reel@localhost:5433/reel_test uv run pytest -q
```

## Gotchas

- **TMDB rate limit: 40 requests / 10 s.** Default `REEL_TMDB_REQUESTS_PER_SECOND=3.5`, burst 1.
  A full 12k seed takes about an hour; it's resumable (`only_missing`). Discover results are
  cached per day under `files/tmdb/discover/`.
- **TMDB auth:** use the *API Read Access Token* (v4, `eyJ…`) as `Authorization: Bearer`. In
  Claude Code cloud sessions the network proxy injects it for `api.themoviedb.org`, so
  `REEL_TMDB_TOKEN` is unset there by design.
- **files.grouplens.org TLS:** its chain roots at *emSign Root CA - G1*, which isn't in this
  container's trust store. Download MovieLens locally and use `reel movielens --zip`, or set
  `REEL_MOVIELENS_CA_BUNDLE`. Never disable verification.
- **Letterboxd:** most pages sit behind Cloudflare (403). RSS (`/{user}/rss/`) works and
  includes `tmdb:movieId`. The terms prohibit scraping; stick to RSS and the user's own
  export, at about 1 request/s.
- **DuckDB 2.x:** relations no longer have `fetchone()`. Use `lake.query(sql)` (eager cursor)
  for fetching and `lake.sql(sql)` (lazy relation) to compose or hand to `write_table`.
- **Dagster:** don't use `from __future__ import annotations` in `definitions.py`, because
  Dagster inspects the `context` annotation at runtime.
- **sklearn pickling:** no lambdas in fitted objects (e.g. the TF-IDF analyzer is the
  module-level `_identity`).
- **pgvector:** `catalog.movie_embedding.embedding` is unconstrained `vector`, so several models
  and dims can coexist. Exact scan is fine at about 10⁴ rows; add per-model HNSW expression
  indexes past about 10⁵.
- **Raw TMDB id exports** are about 28 MB/day; only the last 7 are kept (`prune_exports`).
- **Docker in cloud sessions:** the daemon isn't running at start; `dockerd &` first, then
  `make up`.

## Data sources

**TMDB**
- Getting started: https://developer.themoviedb.org/docs/getting-started
- API reference: https://developer.themoviedb.org/reference/intro/getting-started
- Authentication (Bearer token): https://developer.themoviedb.org/docs/authentication-application
- Rate limiting: https://developer.themoviedb.org/docs/rate-limiting
- Daily ID exports: https://developer.themoviedb.org/docs/daily-id-exports
- Movie details: https://developer.themoviedb.org/reference/movie-details
- `append_to_response`: https://developer.themoviedb.org/docs/append-to-response
- Changes feed: https://developer.themoviedb.org/reference/changes-movie-list
- Discover: https://developer.themoviedb.org/reference/discover-movie
- Images (poster URLs): https://developer.themoviedb.org/docs/image-basics
- Terms of use (attribution, non-commercial): https://www.themoviedb.org/api-terms-of-use

**MovieLens (GroupLens)**
- Datasets: https://grouplens.org/datasets/movielens/
- ml-25m README (genome files): https://files.grouplens.org/datasets/movielens/ml-25m-README.html
- Tag Genome 2021: https://grouplens.org/datasets/movielens/tag-genome-2021/
- Citation, Harper & Konstan 2015: https://doi.org/10.1145/2827872
- Tag Genome paper, Vig, Sen & Riedl 2012: https://doi.org/10.1145/2362394.2362395

**Letterboxd**
- Export your data: https://letterboxd.com/settings/data/
- Import/export format help: https://letterboxd.com/about/importing-data/
- API (approval-gated): https://letterboxd.com/api-beta/
- Terms of use: https://letterboxd.com/legal/terms-of-use/

**Wikidata (planned)**
- SPARQL service: https://query.wikidata.org/
- Docs: https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service

## Stack

**Python & tooling**
- uv: https://docs.astral.sh/uv/
- ruff: https://docs.astral.sh/ruff/
- pytest: https://docs.pytest.org/
- typer: https://typer.tiangolo.com/
- pydantic-settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- httpx: https://www.python-httpx.org/
- tenacity: https://tenacity.readthedocs.io/

**Storage & warehouse**
- fsspec: https://filesystem-spec.readthedocs.io/
- s3fs: https://s3fs.readthedocs.io/
- PyArrow Parquet: https://arrow.apache.org/docs/python/parquet.html
- DuckDB docs: https://duckdb.org/docs/
  - Python API: https://duckdb.org/docs/stable/clients/python/overview
  - Parquet: https://duckdb.org/docs/stable/data/parquet/overview
  - JSON: https://duckdb.org/docs/stable/data/json/overview
  - S3 via httpfs: https://duckdb.org/docs/stable/core_extensions/httpfs/s3api
  - Secrets manager: https://duckdb.org/docs/stable/configuration/secrets_manager
- MinIO: https://min.io/docs/minio/linux/index.html

**Serving**
- PostgreSQL 16: https://www.postgresql.org/docs/16/
- pgvector: https://github.com/pgvector/pgvector
- pg_trgm: https://www.postgresql.org/docs/16/pgtrgm.html
- psycopg 3: https://www.psycopg.org/psycopg3/docs/
  - COPY: https://www.psycopg.org/psycopg3/docs/basic/copy.html
- FastAPI: https://fastapi.tiangolo.com/

**Orchestration**
- Dagster: https://docs.dagster.io/
  - Assets: https://docs.dagster.io/guides/build/assets
  - Partitions & backfills: https://docs.dagster.io/guides/build/partitions-and-backfills
  - Resources: https://docs.dagster.io/guides/build/external-resources
  - Schedules: https://docs.dagster.io/guides/automate/schedules
  - dbt integration (if adopted): https://docs.dagster.io/integrations/libraries/dbt
- dbt-duckdb (candidate for silver): https://github.com/duckdb/dbt-duckdb

**ML (classical / OSS)**
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
  - TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
  - TruncatedSVD / LSA: https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis
  - HDBSCAN: https://scikit-learn.org/stable/modules/clustering.html#hdbscan
  - LogisticRegression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- How HDBSCAN works: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
- c-TF-IDF (BERTopic): https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html
- Bradley–Terry model: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
- Planned: implicit (ALS): https://github.com/benfred/implicit
- Planned: sentence-transformers: https://www.sbert.net/

**Environment**
- Claude Code on the web (environments, network, setup scripts):
  https://code.claude.com/docs/en/claude-code-on-the-web

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-09 | Greenfield on a branch; prototype stays on `main` | Prototype was a stepping stone |
| 2026-11-09 | Lake = Parquet on an fsspec ObjectStore; DuckDB as warehouse | Same code on disk/MinIO/S3; zero infra in dev |
| 2026-11-09 | CDC via source feeds + content hashing + SCD2-from-bronze | Cheapest correct option per source; no Debezium needed |
| 2026-11-09 | Postgres + pgvector for serving; `app.event` outbox | One DB for catalog, vectors and user events |
| 2026-11-09 | Dagster over Airflow/Prefect | Asset-centric model maps to lake layers; good local UI |
| 2026-11-09 | TF-IDF→LSA baseline before neural embeddings | Deterministic, explainable, seconds on CPU |
| 2026-11-09 | TMDB 3.5 req/s, burst 1 | Provider limit is 40 req / 10 s |

## Current state (keep updated)

- Dev catalog: about 10.3k films (top 10k by votes, plus popular). Silver build about 6 s,
  LSA train about 7 s, clustering about 6 s, publish about 3 s.
- Known model weaknesses: neighbors skew toward same era + genre; mood search favors
  keyword-rich franchise films; about 46% HDBSCAN noise. Next: collaborative signal
  (MovieLens ALS), quality prior in mood ranking, Tag Genome mood axes.
- Not built yet: UI, app-events → lake export, Letterboxd export entity resolution,
  Wikidata, MLflow.
