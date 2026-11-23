"""Read selected members from pinned, single-volume WildRGB-D ZIPs via HTTP ranges."""

import io
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, crc32

import requests

REVISION = "58f2b64c8cf5363e08db61ce5111936b6e767b17"
BASE = f"https://huggingface.co/hongchi/wildrgbd/resolve/{REVISION}"
TAIL_SIZE = 65536 + 128


def write_bytes(path, data):
    """Publish a complete cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


class RangeFile:
    """Seekable remote ZIP with cached directory and one in-memory member window."""

    def __init__(self, session, url, cache):
        """Resolve an unrestricted signed download URL without receiving the archive."""
        self.session, self.url, self.cache = session, url, cache
        self.position = self.window_start = 0
        self.window = b""
        self.size = self.connect()
        self.tail_start = max(0, self.size - TAIL_SIZE)
        self.tail = self.cached_range(self.tail_start, self.size - self.tail_start)

    def connect(self):
        """Resolve the immutable source and discover its exact total size."""
        with self.session.head(self.url, allow_redirects=True, timeout=(15, 60)) as response:
            if response.status_code != 200:
                raise OSError(f"WildRGB-D HEAD returned HTTP {response.status_code}")
            self.endpoint = response.url
            return int(response.headers["Content-Length"])

    def fetch(self, start, length):
        """Require an exact 206 before reading any bytes from the object."""
        stop = start + length - 1
        expected = f"bytes {start}-{stop}/{self.size}"
        for attempt in range(5):
            try:
                with self.session.get(
                    self.endpoint,
                    headers={"Range": f"bytes={start}-{stop}"},
                    stream=True,
                    timeout=(15, 120),
                ) as response:
                    if response.status_code in (401, 403):
                        if self.connect() != self.size:
                            raise ValueError("The remote archive size changed")
                        raise OSError("Signed WildRGB-D URL needs renewal")
                    if response.status_code in (408, 429, 500, 502, 503, 504):
                        raise OSError(f"Temporary HTTP {response.status_code}")
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise ValueError(f"WildRGB-D server did not honor {expected}")
                    data = response.content
                    if len(data) != length:
                        raise OSError("Incomplete WildRGB-D range response")
                    return data
            except (OSError, requests.RequestException):
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("Unreachable")

    def cached_range(self, start, length):
        """Persist the central directory and end records for later preparation runs."""
        path = self.cache / f"range-{start}-{length}.bin"
        if path.is_file() and path.stat().st_size == length:
            return path.read_bytes()
        data = self.fetch(start, length)
        write_bytes(path, data)
        return data

    def seek(self, offset, whence=os.SEEK_SET):
        """Expose original absolute ZIP offsets to the standard ZIP64 reader."""
        self.position = (0, self.position, self.size)[whence] + offset
        if self.position < 0:
            raise OSError("Negative ZIP offset")
        return self.position

    def tell(self):
        """Return the current absolute ZIP offset."""
        return self.position

    def seekable(self):
        """Support ZipFile's shared-file wrapper."""
        return True

    def prefetch(self, start, length):
        """Fetch the local header, filename, extra fields and compressed member together."""
        self.window_start = start
        self.window = self.fetch(start, min(length, self.size - start))

    def read(self, size=-1):
        """Serve ZipFile's directory and member reads from the relevant range."""
        start = self.position
        end = self.size if size < 0 else min(start + size, self.size)
        if start >= end:
            return b""
        self.position = end
        if start >= self.tail_start:
            return self.tail[start - self.tail_start : end - self.tail_start]
        if self.window_start <= start and end <= self.window_start + len(self.window):
            return self.window[start - self.window_start : end - self.window_start]
        if end >= self.tail_start:
            # ZipFile's single large read of the central directory.
            return self.cached_range(start, end - start)
        return self.fetch(start, end - start)


class MemberCache:
    """Expose validated ZIP members as bytes or binary in-memory streams."""

    def __init__(self, archive, remote, directory):
        """Keep one archive and cache directory per category."""
        self.archive, self.remote, self.directory = archive, remote, directory

    def read(self, name):
        """Read a named member and check its uncompressed size and CRC32."""
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
            raise ValueError(f"Invalid archive member name: {name}")
        info = self.archive.getinfo(name)
        path = self.directory.joinpath(*rel.parts)
        if path.is_file():
            data = path.read_bytes()
            if len(data) == info.file_size and crc32(data) == info.CRC:
                return data
        header_estimate = 30 + len(name.encode("utf-8")) + max(512, len(info.extra))
        self.remote.prefetch(info.header_offset, header_estimate + info.compress_size)
        data = self.archive.read(info)
        if len(data) != info.file_size or crc32(data) != info.CRC:
            raise BadZipFile(f"Invalid size or CRC for {name}")
        write_bytes(path, data)
        return data

    def open(self, name):
        """Return an independent file-like stream for Pillow."""
        return io.BytesIO(self.read(name))

    def link_to(self, name, destination):
        """Share the verified member bytes with a frame directory on the same volume."""
        data = self.read(name)
        source = self.directory.joinpath(*PurePosixPath(name).parts)
        destination = Path(destination)
        if destination.exists():
            if not source.samefile(destination) and destination.read_bytes() != data:
                raise ValueError(f"Conflicting prepared frame: {destination}")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.hardlink_to(source)
        except OSError:
            pending = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, pending)
            pending.replace(destination)


@contextmanager
def open_archive(category, cache_dir):
    """Open one category without storing its full ZIP or downloading unrelated scenes."""
    if not category.isascii() or not category.replace("_", "").isalnum():
        raise ValueError("Invalid category")
    root = Path(cache_dir) / REVISION / category
    with requests.Session() as session:
        session.headers.update({"Accept-Encoding": "identity"})
        remote = RangeFile(session, f"{BASE}/{category}.zip", root / "index")
        with ZipFile(remote) as archive:
            yield MemberCache(archive, remote, root / "members")
