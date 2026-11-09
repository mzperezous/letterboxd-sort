-- Serving schema. `catalog` is derived (rebuilt from the lake by `reel db publish`);
-- `app` is operational and source-of-truth (user actions). Never write app data to catalog.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE catalog.movie (
    movie_id      BIGINT PRIMARY KEY,
    title         TEXT NOT NULL,
    original_title TEXT,
    release_year  INT,
    runtime_min   INT,
    overview      TEXT,
    poster_path   TEXT,
    vote_average  DOUBLE PRECISION,
    vote_count    INT,
    imdb_id       TEXT,
    genres        TEXT[] NOT NULL DEFAULT '{}',
    keywords      TEXT[] NOT NULL DEFAULT '{}',
    directors     TEXT[] NOT NULL DEFAULT '{}',
    synced_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX movie_title_trgm ON catalog.movie USING gin (title gin_trgm_ops);

-- Unconstrained `vector` so several models (different dims) can coexist. At catalog scale
-- (~1e4-1e5 rows) exact scans are fast; add per-model HNSW expression indexes when needed.
CREATE TABLE catalog.movie_embedding (
    movie_id  BIGINT NOT NULL REFERENCES catalog.movie ON DELETE CASCADE,
    model     TEXT NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (movie_id, model)
);

CREATE TABLE catalog.movie_neighbor (
    movie_id    BIGINT NOT NULL,
    model       TEXT NOT NULL,
    rank        INT NOT NULL,
    neighbor_id BIGINT NOT NULL,
    score       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (movie_id, model, rank)
);

CREATE TABLE catalog.movie_cluster (
    movie_id   BIGINT NOT NULL,
    model      TEXT NOT NULL,
    cluster_id INT NOT NULL,
    PRIMARY KEY (movie_id, model)
);

CREATE TABLE catalog.cluster_profile (
    cluster_id INT PRIMARY KEY,
    size       INT NOT NULL,
    top_terms  TEXT[] NOT NULL,
    exemplars  TEXT[] NOT NULL
);

CREATE TABLE app.app_user (
    user_id             BIGSERIAL PRIMARY KEY,
    username            TEXT UNIQUE NOT NULL,
    letterboxd_username TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Append-only event log of everything users do. This is the app's change stream: the
-- pipeline reads it incrementally by event_id (a monotonically increasing watermark).
CREATE TABLE app.event (
    event_id   BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES app.app_user,
    kind       TEXT NOT NULL,   -- comparison | rating | skip | double_feature_saved | ...
    payload    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX event_user_kind ON app.event (user_id, kind, event_id);

-- Convenience projection of comparison events.
CREATE VIEW app.comparison AS
SELECT event_id, user_id,
       (payload->>'winner_id')::BIGINT AS winner_id,
       (payload->>'loser_id')::BIGINT  AS loser_id,
       created_at
FROM app.event WHERE kind = 'comparison';
