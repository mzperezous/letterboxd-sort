"""HTTP API.

Reads the catalog from Postgres and loads model artifacts from the object store (the same way a
deployed service would pull a model from S3). Writes user actions to the `app.event` log.

Run:  uv run uvicorn reel.api.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import numpy as np
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from reel.config import get_settings
from reel.lake import Lake
from reel.ml import content, taste
from reel.ml.pairs import DoubleFeatureScorer
from reel.storage import store_from_settings

POSTER_BASE = "https://image.tmdb.org/t/p/w342"


@lru_cache
def lake() -> Lake:
    s = get_settings()
    return Lake(store_from_settings(s), s)


@lru_cache
def model() -> content.ContentModel:
    return content.load(lake())


@lru_cache
def scorer() -> DoubleFeatureScorer:
    return DoubleFeatureScorer(lake(), model())


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    model.cache_clear()
    scorer.cache_clear()


app = FastAPI(title="reel", lifespan=lifespan)


def db():
    with psycopg.connect(get_settings().pg_dsn, row_factory=dict_row) as conn:
        yield conn


DB = Annotated[psycopg.Connection, Depends(db)]
MOVIE_COLS = "movie_id, title, release_year, runtime_min, vote_average, vote_count, poster_path, genres, directors"


def _hydrate(conn, ids: list[int]) -> dict[int, dict]:
    rows = conn.execute(
        f"SELECT {MOVIE_COLS} FROM catalog.movie WHERE movie_id = ANY(%s)", [ids]
    ).fetchall()
    for r in rows:
        r["poster_url"] = POSTER_BASE + r["poster_path"] if r["poster_path"] else None
    return {r["movie_id"]: r for r in rows}


@app.get("/health")
def health(conn: DB):
    n = conn.execute("SELECT count(*) AS n FROM catalog.movie").fetchone()["n"]
    return {"ok": True, "movies": n}


@app.get("/movies/search")
def search(conn: DB, q: str = Query(min_length=1), limit: int = 20):
    """Title search (trigram similarity, popularity tiebreak)."""
    rows = conn.execute(
        f"""SELECT {MOVIE_COLS}, similarity(title, %(q)s) AS sim FROM catalog.movie
            WHERE title %% %(q)s OR title ILIKE '%%' || %(q)s || '%%'
            ORDER BY sim DESC, vote_count DESC NULLS LAST LIMIT %(n)s""",
        {"q": q, "n": limit},
    ).fetchall()
    return rows


@app.get("/movies/{movie_id}")
def get_movie(conn: DB, movie_id: int):
    row = conn.execute("SELECT * FROM catalog.movie WHERE movie_id = %s", [movie_id]).fetchone()
    if not row:
        raise HTTPException(404)
    return row


@app.get("/movies/{movie_id}/similar")
def similar(conn: DB, movie_id: int, limit: int = 20, model_name: str = content.MODEL_NAME):
    """Nearest neighbors in embedding space, computed by pgvector (cosine distance)."""
    rows = conn.execute(
        """SELECT m.movie_id, 1 - (e.embedding <=> q.embedding) AS score
            FROM catalog.movie_embedding e
            JOIN catalog.movie_embedding q ON q.movie_id = %(id)s AND q.model = %(model)s
            JOIN catalog.movie m ON m.movie_id = e.movie_id
            WHERE e.model = %(model)s AND e.movie_id <> %(id)s
            ORDER BY e.embedding <=> q.embedding LIMIT %(n)s""",
        {"id": movie_id, "model": model_name, "n": limit},
    ).fetchall()
    meta = _hydrate(conn, [r["movie_id"] for r in rows])
    return [{**meta[r["movie_id"]], "score": round(r["score"], 4)} for r in rows]


@app.get("/movies/{movie_id}/double-features")
def double_features(conn: DB, movie_id: int, limit: int = 10):
    """Explainable pair suggestions (heuristic v0; see reel.ml.pairs)."""
    try:
        cands = scorer().candidates(movie_id, k=limit)
    except KeyError as e:
        raise HTTPException(404, "movie not in model") from e
    meta = _hydrate(conn, [c.movie_id for c in cands])
    return [
        {
            **meta.get(c.movie_id, {"movie_id": c.movie_id, "title": c.title}),
            "score": c.score,
            "features": c.features,
            "shared_keywords": c.shared_keywords,
        }
        for c in cands
    ]


@app.get("/mood")
def mood(conn: DB, q: str = Query(min_length=2), limit: int = 20):
    """Free-text mood/theme search via LSA query folding (no LLM)."""
    hits, tokens = model().search_text(q, k=limit)
    meta = _hydrate(conn, [m for m, _ in hits])
    return {
        "matched_tokens": tokens,
        "results": [{**meta[m], "score": round(s, 4)} for m, s in hits if m in meta],
    }


@app.get("/clusters")
def clusters(conn: DB):
    return conn.execute("SELECT * FROM catalog.cluster_profile ORDER BY size DESC").fetchall()


# ----------------------------------------------------------------------------- users
class Comparison(BaseModel):
    winner_id: int
    loser_id: int


def _user_id(conn, username: str) -> int:
    row = conn.execute(
        """INSERT INTO app.app_user (username) VALUES (%s)
           ON CONFLICT (username) DO UPDATE SET username = EXCLUDED.username RETURNING user_id""",
        [username.lower()],
    ).fetchone()
    return row["user_id"]


@app.post("/users/{username}/comparisons", status_code=201)
def add_comparison(conn: DB, username: str, body: Comparison):
    uid = _user_id(conn, username)
    row = conn.execute(
        "INSERT INTO app.event (user_id, kind, payload) VALUES (%s, 'comparison', %s) RETURNING event_id",
        [uid, Jsonb(body.model_dump())],
    ).fetchone()
    conn.commit()
    return row


@app.get("/users/{username}/recommendations")
def recommendations(conn: DB, username: str, limit: int = 20):
    """Fit a Bradley-Terry taste vector from the user's comparisons and rank unseen films."""
    pairs = [
        (r["winner_id"], r["loser_id"])
        for r in conn.execute(
            """SELECT c.winner_id, c.loser_id FROM app.comparison c
               JOIN app.app_user u USING (user_id) WHERE u.username = %s""",
            [username.lower()],
        ).fetchall()
    ]
    m = model()
    try:
        t = taste.fit(m, pairs)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    seen = {x for p in pairs for x in p}
    scores = t.score(m.embeddings)
    order = [int(m.movie_ids[i]) for i in np.argsort(-scores, kind="stable")]
    top = [i for i in order if i not in seen][:limit]
    meta = _hydrate(conn, top)
    return {"pairs": t.n_pairs, "results": [meta[i] for i in top if i in meta]}
