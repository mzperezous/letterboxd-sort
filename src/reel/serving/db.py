"""Postgres access: connections, a tiny forward-only migration runner, and table sync."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from importlib import resources

import psycopg
from psycopg import sql

log = logging.getLogger(__name__)


def connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=False)


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply migrations/NNN_*.sql in order, once each. Returns newly applied names."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS public.schema_migration "
        "(name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    done = {r[0] for r in conn.execute("SELECT name FROM public.schema_migration")}
    files = sorted(
        f for f in resources.files("reel.serving.migrations").iterdir() if f.name.endswith(".sql")
    )
    applied = []
    for f in files:
        if f.name in done:
            continue
        log.info("applying migration %s", f.name)
        conn.execute(f.read_text())
        conn.execute("INSERT INTO public.schema_migration (name) VALUES (%s)", [f.name])
        applied.append(f.name)
    conn.commit()
    return applied


def sync_table(
    conn: psycopg.Connection,
    table: str,
    columns: Sequence[str],
    key: Sequence[str],
    rows: Iterable[Sequence],
    *,
    delete_missing: bool = True,
    scope: tuple[str, object] | None = None,
) -> dict[str, int]:
    """Make `table` match `rows`: COPY into a temp table, upsert, then delete rows not present.

    `scope=(column, value)` limits deletion to one slice (e.g. model = 'content_lsa_v1').
    Runs inside the caller's transaction, so readers see the old or the new state, never a mix.
    """
    schema, name = table.split(".")
    tmp = f"tmp_{name}"
    cols = sql.SQL(", ").join(map(sql.Identifier, columns))
    conn.execute(
        sql.SQL("CREATE TEMP TABLE {} (LIKE {} INCLUDING DEFAULTS) ON COMMIT DROP").format(
            sql.Identifier(tmp), sql.Identifier(schema, name)
        )
    )
    n = 0
    with conn.cursor().copy(
        sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(tmp), cols)
    ) as cp:
        for r in rows:
            cp.write_row(r)
            n += 1
    non_key = [c for c in columns if c not in key]
    conflict = sql.SQL(", ").join(map(sql.Identifier, key))
    update = (
        sql.SQL("DO UPDATE SET ")
        + sql.SQL(", ").join(
            sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c)) for c in non_key
        )
        if non_key
        else sql.SQL("DO NOTHING")
    )
    conn.execute(
        sql.SQL("INSERT INTO {t} ({c}) SELECT {c} FROM {tmp} ON CONFLICT ({k}) ").format(
            t=sql.Identifier(schema, name), c=cols, tmp=sql.Identifier(tmp), k=conflict
        )
        + update
    )
    deleted = 0
    if delete_missing:
        match = sql.SQL(" AND ").join(
            sql.SQL("t.{c} = s.{c}").format(c=sql.Identifier(c)) for c in key
        )
        where = sql.SQL("")
        params: list = []
        if scope:
            where = sql.SQL(" AND t.{} = %s").format(sql.Identifier(scope[0]))
            params = [scope[1]]
        cur = conn.execute(
            sql.SQL("DELETE FROM {t} t WHERE NOT EXISTS (SELECT 1 FROM {tmp} s WHERE {m})").format(
                t=sql.Identifier(schema, name), tmp=sql.Identifier(tmp), m=match
            )
            + where,
            params,
        )
        deleted = cur.rowcount
    conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(tmp)))
    return {"rows": n, "deleted": deleted}
