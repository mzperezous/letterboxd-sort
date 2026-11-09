"""The lake: bronze (raw, append-only), silver (normalized), gold (features/models).

Layout (relative keys inside the ObjectStore):

    bronze/{source}/{entity}/ingest_date=YYYY-MM-DD/{run_id}-{n}.parquet   raw records
    files/{source}/{name}                                                  raw files (zips, dumps)
    silver/{table}/data.parquet                                            rebuilt from bronze
    gold/{table}/data.parquet                                              features, embeddings

Change data capture
-------------------
Every bronze record carries a `content_hash` of its canonical payload. Writers ask the warehouse
for the latest hash per key and only append records whose hash differs, so re-running an ingest
is cheap and idempotent, and the bronze history is a clean change log. Silver SCD2 tables are a
pure function of bronze (see transform.py), so they can always be rebuilt from scratch.

A 404 / deletion is stored as a tombstone record (status=404, payload=null) so deletions flow
through the same change log.
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from reel.config import Settings
from reel.storage import FsspecStore

TOMBSTONE_HASH = "tombstone"

BRONZE_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("entity", pa.string()),
        ("key", pa.string()),
        ("fetched_at", pa.timestamp("us", tz="UTC")),
        ("run_id", pa.string()),
        ("status", pa.int32()),
        ("content_hash", pa.string()),
        ("payload", pa.string()),  # canonical JSON text; null for tombstones
    ]
)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(payload) -> str:
    if payload is None:
        return TOMBSTONE_HASH
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


@dataclass
class RawRecord:
    key: str
    payload: dict | list | None
    status: int = 200
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def hash(self) -> str:
        return content_hash(self.payload)


def to_parquet_bytes(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


class Lake:
    def __init__(self, store: FsspecStore, settings: Settings | None = None):
        self.store = store
        self.settings = settings
        self._con: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------ warehouse (DuckDB)
    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect()
            if self.store.protocol == "s3" and self.settings is not None:
                self._configure_s3(self._con, self.settings)
        return self._con

    @staticmethod
    def _configure_s3(con: duckdb.DuckDBPyConnection, s: Settings) -> None:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        parts = [
            "TYPE S3",
            f"KEY_ID '{s.s3_access_key_id or ''}'",
            f"SECRET '{s.s3_secret_access_key.get_secret_value() if s.s3_secret_access_key else ''}'",
            f"REGION '{s.s3_region}'",
        ]
        if s.s3_endpoint_url:
            endpoint = s.s3_endpoint_url.split("://", 1)[-1]
            parts += [f"ENDPOINT '{endpoint}'", "URL_STYLE 'path'"]
            if s.s3_endpoint_url.startswith("http://"):
                parts.append("USE_SSL false")
        con.execute(f"CREATE OR REPLACE SECRET lake ({', '.join(parts)})")

    def sql(self, query: str, params: list | None = None) -> duckdb.DuckDBPyRelation:
        """Lazy relation (compose further, or hand to write_table)."""
        return self.con.sql(query, params=params)

    def query(self, query: str, params: list | None = None) -> duckdb.DuckDBPyConnection:
        """Execute now; use .fetchall() / .fetchone() / .df() on the result."""
        return self.con.cursor().execute(query, params or [])

    # ------------------------------------------------------------------ bronze
    def bronze_prefix(self, source: str, entity: str) -> str:
        return f"bronze/{source}/{entity}"

    def bronze_exists(self, source: str, entity: str) -> bool:
        return bool(self.store.list(self.bronze_prefix(source, entity)))

    def bronze_glob(self, source: str, entity: str) -> str:
        return self.store.uri(self.bronze_prefix(source, entity)) + "/**/*.parquet"

    def bronze(self, source: str, entity: str) -> str:
        """SQL table expression for a bronze dataset (empty-safe)."""
        if not self.bronze_exists(source, entity):
            cols = ", ".join(f"NULL::{_duck_type(f.type)} AS {f.name}" for f in BRONZE_SCHEMA)
            return f"(SELECT {cols} WHERE false)"
        return f"read_parquet('{self.bronze_glob(source, entity)}', hive_partitioning = true)"

    def latest_hashes(self, source: str, entity: str) -> dict[str, str]:
        if not self.bronze_exists(source, entity):
            return {}
        rows = self.query(
            f"""
            SELECT key, arg_max(content_hash, fetched_at) AS h
            FROM {self.bronze(source, entity)}
            GROUP BY key
            """
        ).fetchall()
        return dict(rows)

    def write_bronze(
        self,
        source: str,
        entity: str,
        records: Iterable[RawRecord],
        *,
        latest: dict[str, str] | None = None,
        run_id: str | None = None,
        chunk_size: int = 2000,
    ) -> dict[str, int]:
        """Append records whose content changed. Returns counts {seen, written, unchanged}.

        `latest` lets callers that write many chunks reuse one hash lookup; it is updated in place.
        """
        if latest is None:
            latest = self.latest_hashes(source, entity)
        run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        stats = {"seen": 0, "written": 0, "unchanged": 0}
        batch: list[RawRecord] = []
        part = 0

        def flush():
            nonlocal part, batch
            if not batch:
                return
            day = batch[0].fetched_at.strftime("%Y-%m-%d")
            table = pa.Table.from_pylist(
                [
                    {
                        "source": source,
                        "entity": entity,
                        "key": r.key,
                        "fetched_at": r.fetched_at,
                        "run_id": run_id,
                        "status": r.status,
                        "content_hash": r.hash,
                        "payload": None if r.payload is None else canonical_json(r.payload),
                    }
                    for r in batch
                ],
                schema=BRONZE_SCHEMA,
            )
            key = f"{self.bronze_prefix(source, entity)}/ingest_date={day}/{run_id}-{part:05d}.parquet"
            self.store.put(key, to_parquet_bytes(table))
            part += 1
            batch = []

        for rec in records:
            stats["seen"] += 1
            h = rec.hash
            if latest.get(rec.key) == h:
                stats["unchanged"] += 1
                continue
            # A tombstone for a key we've never seen is noise.
            if h == TOMBSTONE_HASH and rec.key not in latest:
                stats["unchanged"] += 1
                continue
            latest[rec.key] = h
            batch.append(rec)
            stats["written"] += 1
            if len(batch) >= chunk_size:
                flush()
        flush()
        return stats

    # ------------------------------------------------------------------ files
    def put_file(self, source: str, name: str, data: bytes) -> str:
        key = f"files/{source}/{name}"
        self.store.put(key, data)
        return key

    # ------------------------------------------------------------------ silver / gold tables
    def table_key(self, layer: str, name: str) -> str:
        return f"{layer}/{name}/data.parquet"

    def write_table(self, layer: str, name: str, data: pa.Table | duckdb.DuckDBPyRelation) -> int:
        if isinstance(data, duckdb.DuckDBPyRelation):
            data = data.arrow()
            if isinstance(data, pa.RecordBatchReader):
                data = data.read_all()
        self.store.put(self.table_key(layer, name), to_parquet_bytes(data))
        return data.num_rows

    def has_table(self, layer: str, name: str) -> bool:
        return self.store.exists(self.table_key(layer, name))

    def table(self, layer: str, name: str) -> str:
        """SQL table expression, e.g. FROM {lake.table('silver', 'movie')}."""
        if not self.has_table(layer, name):
            raise LookupError(f"{layer}.{name} has not been built yet")
        return f"read_parquet('{self.store.uri(self.table_key(layer, name))}')"

    def register_views(self) -> list[str]:
        """Expose every built silver/gold table as a DuckDB view named {layer}_{table}."""
        names = []
        for layer in ("silver", "gold"):
            for key in self.store.list(layer):
                if not key.endswith("/data.parquet"):
                    continue
                name = key.split("/")[1]
                view = f"{layer}_{name}"
                self.con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {self.table(layer, name)}"
                )
                names.append(view)
        return names


def _duck_type(t: pa.DataType) -> str:
    if pa.types.is_timestamp(t):
        return "TIMESTAMPTZ"
    if pa.types.is_int32(t):
        return "INTEGER"
    return "VARCHAR"
