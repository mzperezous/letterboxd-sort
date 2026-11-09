"""Object storage abstraction.

Pipelines only ever talk to `ObjectStore` with *relative keys* ("bronze/tmdb/movie/..."). The
concrete backend is chosen from `Settings.lake_url`:

    file:///abs/path   -> local disk (dev)
    s3://bucket/prefix -> S3 or any S3-compatible store (MinIO locally, S3/R2 in prod)
    memory://name      -> in-process (tests)

`uri(key)` returns a fully-qualified location that DuckDB can read directly, which is how the
warehouse layer queries the lake without copying data around.
"""

from __future__ import annotations

from typing import Protocol
from urllib.parse import urlparse

import fsspec

from reel.config import Settings


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...
    def delete(self, key: str) -> None: ...
    def uri(self, key: str) -> str: ...


class FsspecStore:
    """One implementation for every backend fsspec supports."""

    def __init__(self, url: str, storage_options: dict | None = None):
        parsed = urlparse(url)
        self.protocol = parsed.scheme or "file"
        if self.protocol == "file":
            self.root = parsed.path.rstrip("/")
        else:
            self.root = (parsed.netloc + parsed.path).rstrip("/")
        self.fs = fsspec.filesystem(self.protocol, **(storage_options or {}))

    def _path(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.root}/{key}" if key else self.root

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        if self.protocol == "file":
            self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        # Write to a temp key then move, so readers never observe a partial object on local disk.
        if self.protocol == "file":
            tmp = path + ".tmp"
            with self.fs.open(tmp, "wb") as f:
                f.write(data)
            self.fs.mv(tmp, path)
        else:
            with self.fs.open(path, "wb") as f:
                f.write(data)

    def get(self, key: str) -> bytes:
        with self.fs.open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return self.fs.exists(self._path(key))

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not self.fs.exists(base):
            return []
        root = self.root.lstrip("/")
        out = []
        for p in self.fs.find(base):
            q = p.lstrip("/")
            rel = q[len(root) :].lstrip("/") if q.startswith(root) else q
            if not rel.endswith(".tmp"):
                out.append(rel)
        return sorted(out)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if self.fs.exists(path):
            self.fs.rm(path, recursive=True)

    def uri(self, key: str) -> str:
        path = self._path(key)
        if self.protocol == "file":
            return path
        return f"{self.protocol}://{path}"


def store_from_settings(settings: Settings) -> FsspecStore:
    options: dict = {}
    if settings.is_s3:
        options = {
            "key": settings.s3_access_key_id,
            "secret": settings.s3_secret_access_key.get_secret_value()
            if settings.s3_secret_access_key
            else None,
            "client_kwargs": {"region_name": settings.s3_region},
        }
        if settings.s3_endpoint_url:
            options["client_kwargs"]["endpoint_url"] = settings.s3_endpoint_url
    return FsspecStore(settings.lake_url, options)
