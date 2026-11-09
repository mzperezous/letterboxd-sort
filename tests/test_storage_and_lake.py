from datetime import UTC, datetime, timedelta

import pytest

from reel.lake import TOMBSTONE_HASH, Lake, RawRecord, content_hash, scd2_sql
from reel.storage import FsspecStore


@pytest.mark.parametrize("url", ["memory://reel-test", "file://{tmp}"])
def test_store_roundtrip(tmp_path, url):
    store = FsspecStore(url.format(tmp=tmp_path / "s"))
    store.put("a/b/c.txt", b"hello")
    store.put("a/d.txt", b"world")
    assert store.get("a/b/c.txt") == b"hello"
    assert store.exists("a/d.txt") and not store.exists("a/zzz")
    assert store.list("a") == ["a/b/c.txt", "a/d.txt"]
    store.delete("a/b")
    assert store.list("a") == ["a/d.txt"]
    assert store.list("missing") == []


def test_content_hash_is_order_independent():
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert content_hash(None) == TOMBSTONE_HASH


def test_write_bronze_only_appends_changes(lake: Lake):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    s1 = lake.write_bronze(
        "src",
        "thing",
        [RawRecord("1", {"v": 1}, fetched_at=t0), RawRecord("2", {"v": 1}, fetched_at=t0)],
    )
    assert s1 == {"seen": 2, "written": 2, "unchanged": 0}

    t1 = t0 + timedelta(days=1)
    s2 = lake.write_bronze(
        "src",
        "thing",
        [RawRecord("1", {"v": 1}, fetched_at=t1), RawRecord("2", {"v": 2}, fetched_at=t1)],
    )
    assert s2 == {"seen": 2, "written": 1, "unchanged": 1}

    # Tombstone for a known key is written; for an unknown key it's ignored.
    t2 = t1 + timedelta(days=1)
    s3 = lake.write_bronze(
        "src",
        "thing",
        [
            RawRecord("1", None, status=404, fetched_at=t2),
            RawRecord("999", None, status=404, fetched_at=t2),
        ],
    )
    assert s3["written"] == 1

    rows = lake.query(
        f"SELECT key, version, is_current, is_deleted FROM ({scd2_sql(lake.bronze('src', 'thing'))}) ORDER BY key, version"
    ).fetchall()
    assert rows == [
        ("1", 1, False, False),
        ("1", 2, True, True),
        ("2", 1, False, False),
        ("2", 2, True, False),
    ]


def test_scd2_collapses_flip_flops(lake: Lake):
    """A -> B -> A is three versions (history matters), but A -> A is one."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    for i, v in enumerate([1, 2, 2, 1]):
        # bypass dedupe to simulate bronze that contains repeats
        lake.write_bronze(
            "src", "x", [RawRecord("k", {"v": v}, fetched_at=t + timedelta(days=i))], latest={}
        )
    rows = lake.query(
        f"SELECT version, payload, valid_to IS NULL FROM ({scd2_sql(lake.bronze('src', 'x'))}) ORDER BY version"
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [(1, '{"v":1}'), (2, '{"v":2}'), (3, '{"v":1}')]
    assert [r[2] for r in rows] == [False, False, True]


def test_empty_bronze_is_queryable(lake: Lake):
    assert lake.latest_hashes("nope", "nothing") == {}
    assert lake.query(f"SELECT count(*) FROM {lake.bronze('nope', 'nothing')}").fetchone() == (0,)
