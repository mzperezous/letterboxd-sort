"""Content embeddings: field-weighted TF-IDF -> LSA (TruncatedSVD) -> L2-normalized vectors.

Why this first: it's deterministic (fixed random_state), cheap (seconds on CPU), explainable (every
dimension is a mix of real tokens), and a strong baseline for "movies like this".

Each film becomes a bag of typed tokens from several fields:

    genre:science_fiction  kw:time_travel  dir:agnes_varda  cast:toshiro_mifune
    dop:roger_deakins  music:ennio_morricone  lang:ja  country:FR  decade:1970s  coll:alien

Each field gets its own TfidfVectorizer (so a common keyword can't drown a rare director) and a
weight, then the blocks are stacked and reduced with SVD. This is classic Latent Semantic
Analysis; similar directors/keywords end up near each other because they co-occur in films.

Text queries ("lonely space melancholy") can be folded into the same space: match query words to
keyword vocabulary, build a TF-IDF vector, project with the fitted SVD, then cosine-rank.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import pyarrow as pa
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from reel.lake import Lake

MODEL_NAME = "content_lsa_v1"

# Field -> weight. Tuning these is one of the easiest, most informative experiments to run.
FIELD_WEIGHTS: dict[str, float] = {
    "kw": 1.0,
    "genre": 0.6,
    "dir": 0.8,
    "writer": 0.4,
    "cast": 0.5,
    "dop": 0.3,
    "music": 0.3,
    "lang": 0.3,
    "country": 0.2,
    "decade": 0.3,
    "coll": 0.6,
}

# TMDB keywords that describe the release/production, not the film's content.
KEYWORD_STOPLIST = {
    "duringcreditsstinger",
    "aftercreditsstinger",
    "post_credits_scene",
    "sequel",
    "remake",
    "3d",
    "imax",
    "live_action_remake",
}

CREW_FIELDS = {
    "Director": "dir",
    "Screenplay": "writer",
    "Writer": "writer",
    "Director of Photography": "dop",
    "Original Music Composer": "music",
}


def _identity(tokens: list[str]) -> list[str]:
    """Analyzer for pre-tokenized docs (module-level so the model pickles)."""
    return tokens


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def build_documents(lake: Lake, min_votes: int = 0) -> dict[int, dict[str, list[str]]]:
    """movie_id -> {field: [tokens]} from silver tables."""
    t = lambda n: lake.table("silver", n)  # noqa: E731
    movies = lake.query(
        f"""SELECT movie_id, original_language, release_year, collection_id
            FROM {t("movie")} WHERE coalesce(vote_count, 0) >= ?""",
        params=[min_votes],
    ).fetchall()
    docs: dict[int, dict[str, list[str]]] = {}
    for mid, lang, year, coll in movies:
        d: dict[str, list[str]] = {f: [] for f in FIELD_WEIGHTS}
        if lang:
            d["lang"].append(lang)
        if year:
            d["decade"].append(f"{year // 10 * 10}s")
        if coll:
            d["coll"].append(str(coll))
        docs[mid] = d

    def add(sql: str, fld_of=None):
        for mid, fld, val in lake.query(sql).fetchall():
            if mid in docs and val:
                tok = slug(val)
                if fld == "kw" and tok in KEYWORD_STOPLIST:
                    continue
                docs[mid][fld_of(fld) if fld_of else fld].append(tok)

    add(f"SELECT movie_id, 'genre', genre FROM {t('movie_genre')}")
    add(f"SELECT movie_id, 'kw', keyword FROM {t('movie_keyword')}")
    add(f"SELECT movie_id, 'cast', name FROM {t('movie_cast')} WHERE billing_order < 6")
    add(f"SELECT movie_id, 'country', country_code FROM {t('movie_country')}")
    jobs = ", ".join(f"'{j}'" for j in CREW_FIELDS)
    add(
        f"SELECT movie_id, job, name FROM {t('movie_crew')} WHERE job IN ({jobs})",
        fld_of=CREW_FIELDS.get,
    )
    return docs


@dataclass
class ContentModel:
    movie_ids: np.ndarray
    vectorizers: dict[str, TfidfVectorizer]
    svd: TruncatedSVD
    embeddings: np.ndarray  # (n, k), L2-normalized
    weights: dict[str, float] = field(default_factory=lambda: dict(FIELD_WEIGHTS))
    name: str = MODEL_NAME

    # -------------------------------------------------------------- fit / transform
    @classmethod
    def fit(
        cls, docs: dict[int, dict[str, list[str]]], dim: int = 128, seed: int = 0
    ) -> ContentModel:
        ids = np.array(sorted(docs))
        vectorizers, blocks = {}, []
        for fld, w in FIELD_WEIGHTS.items():
            corpus = [docs[i][fld] for i in ids]
            if not any(corpus):
                continue
            vec = TfidfVectorizer(
                analyzer=_identity,
                min_df=2 if fld not in ("lang", "decade", "genre") else 1,
                sublinear_tf=True,
                norm="l2",
            )
            try:
                X = vec.fit_transform(corpus)
            except ValueError:  # empty vocabulary after min_df
                continue
            vectorizers[fld] = vec
            blocks.append(X * w)
        X = sparse.hstack(blocks).tocsr()
        k = max(2, min(dim, X.shape[1] - 1, X.shape[0] - 1))
        svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=7, random_state=seed)
        Z = normalize(svd.fit_transform(X))
        return cls(ids, vectorizers, svd, Z.astype(np.float32))

    def _stack(self, doc_fields: list[dict[str, list[str]]]) -> sparse.csr_matrix:
        blocks = [
            vec.transform([d.get(fld, []) for d in doc_fields]) * self.weights[fld]
            for fld, vec in self.vectorizers.items()
        ]
        return sparse.hstack(blocks).tocsr()

    def embed(self, doc_fields: list[dict[str, list[str]]]) -> np.ndarray:
        return normalize(self.svd.transform(self._stack(doc_fields))).astype(np.float32)

    # -------------------------------------------------------------- queries
    def index_of(self, movie_id: int) -> int:
        idx = np.searchsorted(self.movie_ids, movie_id)
        if idx >= len(self.movie_ids) or self.movie_ids[idx] != movie_id:
            raise KeyError(movie_id)
        return int(idx)

    def similar(self, movie_id: int, k: int = 10) -> list[tuple[int, float]]:
        q = self.embeddings[self.index_of(movie_id)]
        return self._rank(q, k, exclude={movie_id})

    def _rank(
        self, q: np.ndarray, k: int, exclude: set[int] = frozenset()
    ) -> list[tuple[int, float]]:
        scores = self.embeddings @ q
        order = np.argsort(-scores, kind="stable")
        out = []
        for i in order:
            mid = int(self.movie_ids[i])
            if mid in exclude:
                continue
            out.append((mid, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def query_tokens(self, text: str) -> dict[str, list[str]]:
        """Map free text to known keyword/genre tokens (unigrams..trigrams)."""
        words = re.findall(r"[a-z0-9]+", text.lower())
        grams = {"_".join(words[i : i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
        fields: dict[str, list[str]] = {}
        for fld in ("kw", "genre"):
            vec = self.vectorizers.get(fld)
            if vec is not None:
                fields[fld] = sorted(g for g in grams if g in vec.vocabulary_)
        return fields

    def search_text(self, text: str, k: int = 20) -> tuple[list[tuple[int, float]], dict]:
        toks = self.query_tokens(text)
        if not any(toks.values()):
            return [], toks
        return self._rank(self.embed([toks])[0], k), toks

    # -------------------------------------------------------------- neighbors table
    def neighbors(self, k: int = 50, chunk: int = 2048) -> pa.Table:
        """Exact top-k cosine neighbors for every movie (blockwise matmul)."""
        n = len(self.movie_ids)
        k = min(k, n - 1)
        src, dst, rank, score = [], [], [], []
        for start in range(0, n, chunk):
            S = self.embeddings[start : start + chunk] @ self.embeddings.T
            for r in range(S.shape[0]):
                S[r, start + r] = -np.inf
            top = np.argpartition(-S, k, axis=1)[:, :k]
            for r in range(S.shape[0]):
                idx = top[r][np.argsort(-S[r, top[r]], kind="stable")]
                src += [int(self.movie_ids[start + r])] * k
                dst += self.movie_ids[idx].tolist()
                rank += list(range(1, k + 1))
                score += S[r, idx].astype(float).tolist()
        return pa.table(
            {
                "movie_id": src,
                "neighbor_id": dst,
                "rank": rank,
                "score": score,
                "model": [self.name] * len(src),
            }
        )

    def embeddings_table(self) -> pa.Table:
        return pa.table(
            {
                "movie_id": self.movie_ids.tolist(),
                "model": [self.name] * len(self.movie_ids),
                "embedding": pa.array(self.embeddings.tolist(), type=pa.list_(pa.float32())),
            }
        )

    # -------------------------------------------------------------- persistence
    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        joblib.dump(self, buf, compress=3)
        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> ContentModel:
        return joblib.load(io.BytesIO(data))


def model_key(name: str = MODEL_NAME) -> str:
    return f"models/{name}/model.joblib"


def train(lake: Lake, dim: int = 128, min_votes: int = 0, neighbors_k: int = 50) -> dict:
    docs = build_documents(lake, min_votes=min_votes)
    model = ContentModel.fit(docs, dim=dim)
    lake.store.put(model_key(), model.to_bytes())
    lake.write_table("gold", "movie_embedding", model.embeddings_table())
    nn = model.neighbors(k=neighbors_k)
    lake.write_table("gold", "movie_neighbor", nn)
    return {
        "model": model.name,
        "movies": len(model.movie_ids),
        "dim": model.embeddings.shape[1],
        "explained_variance": round(float(model.svd.explained_variance_ratio_.sum()), 4),
        "vocab": {f: len(v.vocabulary_) for f, v in model.vectorizers.items()},
        "neighbor_rows": nn.num_rows,
    }


def load(lake: Lake, name: str = MODEL_NAME) -> ContentModel:
    return ContentModel.from_bytes(lake.store.get(model_key(name)))
