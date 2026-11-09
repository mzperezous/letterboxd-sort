# Architecture

## Goals

1. A film database that stays current without full reloads.
2. Reusable pipeline + ML infrastructure: dev on a laptop, the same code in production.
3. Two products on top: **double-feature finder** and **mood-based search**, later
   personalized from Letterboxd profiles and in-app rankings.

## Data flow

```
                  ┌────────────── sources ──────────────┐
                  │ TMDB export  TMDB API   MovieLens   │
                  │ Letterboxd RSS / export  (Wikidata) │
                  └──────────────────┬──────────────────┘
                                     │  async httpx, token-bucket rate limit, retries
                                     ▼
 ObjectStore  ┌─────────────────────────────────────────────────────────────┐
 (file:// or  │ bronze/{source}/{entity}/ingest_date=D/*.parquet  append-only│
  s3://)      │ files/{source}/...                                raw dumps  │
              │ silver/{table}/data.parquet      rebuilt from bronze (DuckDB)│
              │ gold/{table}/data.parquet        features, embeddings, kNN   │
              │ models/{name}/model.joblib       fitted model artifacts      │
              └───────────────────────┬─────────────────────────────────────┘
                                      │ reel db publish (one transaction)
                                      ▼
                        Postgres: catalog.* (read model, pgvector)
                                  app.*     (source of truth for user actions)
                                      │
                                      ▼
                              FastAPI  →  UI (next)
```

Rules that keep this sane:

- **Bronze is immutable and append-only.** Nothing ever rewrites it. All history lives there.
- **Silver and gold are pure functions of bronze.** Any table can be deleted and rebuilt. This
  makes backfills, schema changes and bug fixes boring.
- **Postgres `catalog` is disposable.** It's a projection of the lake for low-latency serving.
- **Postgres `app` is the opposite:** user actions originate there and flow *back* into the
  lake as an event stream (planned: `bronze/app/event`, read incrementally by `event_id`).

## Change data capture

There is no single CDC mechanism; each source gets the cheapest one that's correct.

| Source | How changes are detected | Stored as |
|---|---|---|
| TMDB id universe | Diff of consecutive daily id exports (new ids; missing ids → tombstones) | `bronze/tmdb/id_export` |
| TMDB movie data | `/movie/changes` feed (daily) ∩ our catalog → re-fetch → content hash compare | `bronze/tmdb/movie` |
| TMDB popularity | Raw daily export file (popularity is too volatile to version) | `files/tmdb/id_exports/` |
| MovieLens | Static, versioned dumps; re-ingest only on a new release | `files/movielens/{variant}/` |
| Letterboxd export | Re-import → row-level content hash → only changed ratings/watches append | `bronze/letterboxd/export_*` |
| Letterboxd RSS | Poll; GUID-keyed, hash-deduped | `bronze/letterboxd/rss_entry` |
| App (Postgres) | Append-only `app.event` table, monotonically increasing `event_id` watermark (outbox pattern) | planned |

**Content hashing** (`reel.lake.content_hash`) is SHA-256 of canonical JSON. Volatile fields
are stripped *before* hashing (`tmdb.trim_movie` drops `popularity`), so a "version" means a
meaningful change. `Lake.write_bronze` looks up the latest hash per key and only appends
changes, so every ingest is idempotent: re-run it freely.

**SCD2** (`reel.transform.scd2_sql`) turns a bronze change log into versions with
`valid_from / valid_to / is_current / is_deleted`. Because it's computed from bronze each time,
there's no merge logic to get wrong. At ~10⁵ films a full rebuild is seconds; past ~10⁷ rows
you'd switch the same SQL to incremental-by-partition.

Why not Debezium/logical replication? That's for capturing changes in a database *you own* at
high volume. The only DB we own is the app DB, and an outbox table gives the same guarantees
with zero infrastructure until traffic justifies more.

## Storage abstraction

`reel.storage.ObjectStore` is the only interface pipelines see: `put/get/exists/list/delete`
plus `uri(key)`, which returns a location DuckDB can read directly. `FsspecStore` implements it
for `file://`, `s3://` (MinIO, S3, R2, GCS-interop) and `memory://` (tests). Choosing a
backend is just `REEL_LAKE_URL`.

DuckDB reads Parquet straight from the store (`httpfs` + an S3 secret for `s3://`), so the
warehouse is "DuckDB over the lake" in dev and can become MotherDuck / Athena / BigQuery
external tables later without changing the lake format.

## Orchestration

Dagster software-defined assets (`reel.pipelines.definitions`) mirror the lake layers. Every
asset calls the same library function as the CLI, so there's one code path. `tmdb_changes` is
daily-partitioned, so a missed day is a backfill of one partition.

- `bootstrap` job: everything except the change feed.
- `daily_refresh` job + schedule (09:17 UTC, after TMDB publishes exports): id export, changes,
  RSS, then everything downstream.

## Dev → prod

| Concern | Dev | Prod (no code change) |
|---|---|---|
| Lake | `file://./data/lake` or MinIO | S3 / R2 / GCS (`REEL_LAKE_URL=s3://…`) |
| Warehouse | DuckDB in-process | DuckDB in jobs (fine to ~100 GB), or MotherDuck |
| Serving DB | Postgres+pgvector in Docker | Managed Postgres with pgvector (Neon, Supabase, RDS, Cloud SQL) |
| Orchestration | `dagster dev` | Dagster+ Serverless/Hybrid, or dagster on ECS/k8s |
| API | `uvicorn --reload` | Container (Fly, Cloud Run, ECS) reading models from the object store |
| Secrets | `.env` / proxy injection | Platform secret manager → `REEL_*` env vars |

## Modeling roadmap

All deterministic and open source first; LLMs only where they clearly earn it.

1. **Content embeddings: done.** Field-weighted TF-IDF → LSA (TruncatedSVD). Exact kNN.
2. **Clusters: done.** PCA → HDBSCAN → c-TF-IDF cluster profiles.
3. **Double features v0: done.** Interpretable pair features + hand weights.
4. **Taste v0: done.** Bradley-Terry (logistic regression on embedding differences).
5. **Collaborative embeddings.** Implicit ALS / item2vec on MovieLens ratings (and Letterboxd
   list co-occurrence): "people who love X love Y". Combine with content (concatenate, or
   learn weights).
6. **Mood axes.** MovieLens Tag Genome (ml-25m) gives 1,128 tags with dense relevance scores
   for ~14k films. Pick ~40 mood tags as axes, then train ridge regression from our embeddings
   → genome scores to extend moods to films the genome doesn't cover. Evaluate on held-out
   genome films.
7. **Review-text features.** Sentence-transformer embeddings (OSS, CPU-feasible at this scale)
   of RSS review snippets; or TF-IDF + NMF topics as the fully classical option.
8. **Learned pair model.** Collect positive double bills (curated lists, repertory cinema
   programs) → logistic regression / gradient boosting over the pair features in
   `reel.ml.pairs`, negatives sampled from neighbors. Metric: recall@k on held-out pairs.
9. **LLM (limited).** Naming clusters from c-TF-IDF profiles, and writing one-line "why these
   two" blurbs. Both cached, both optional.

## Known gaps / next steps

- **UI** (planned): search, film page with similar + double features, ranking flow (the
  original prototype's Beli-style comparisons), mood search.
- **App events → lake** export asset, so in-app comparisons train taste models offline too.
- **Entity resolution** for Letterboxd export rows (title+year → TMDB id; the RSS already
  carries TMDB ids).
- **Wikidata** source for film movements, awards, and better crosswalks.
- **MLflow** tracking once there's more than one model variant worth comparing.
