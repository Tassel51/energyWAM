#!/usr/bin/env python3
"""Resumable, verified download/extract worker for official RoboCasa365 data.

Consumes the manifest built by
``coordination/full_download_20261030/build_catalog.py`` (``download_manifest.json``,
schema ``energywam_robocasa365_download_v1``) and installs:

* 50 ``target`` human dataset TARs -> ``datasets/robocasa365/v1.0/target/<atomic|composite>/<Task>/<date>/lerobot``
* 6 asset ZIPs -> ``sim/robocasa365/robocasa/models/assets/...`` (see
  :func:`install_asset` for the exact placement rules)

Guarantees (all stdlib, Python 3.11+):

* exclusive ``flock`` so two workers cannot double-install;
* resumable ``<id>.tar.part`` / ``<id>.zip.part`` with strict HTTP
  ``Content-Range`` / size checks; HTML error pages are never accepted as
  archives; SHA-256 recorded on disk next to the archive;
* safe extraction: traversal / absolute paths / drive letters / symlinks /
  hardlinks / device nodes rejected; only ``lerobot/`` is taken from dataset
  TARs (the top-level ``README.md`` is ignored);
* dataset staging + verify counts against ``info.json``, then one atomic
  rename into place; existing user data is never overwritten;
* asset ZIPs are staged per task and merged atomically only into an absent or
  empty target; ``__MACOSX`` / ``.DS_Store`` ignored; a marker file records a
  validated install;
* per-item persistent status JSON + aggregate status; failures are recorded and
  independent items keep going; bounded retries with backoff;
* SIGTERM/SIGINT preserve partial downloads;
* space guard: >=2 GiB free reserve and a ``--max-storage-gib`` ceiling
  (default 200) over what this worker downloads + extracts; entries without a
  known ``archive_bytes`` are reported and not downloaded until the root
  supplies the real total.

CLI::

    python robocasa_download_worker.py \
        --manifest coordination/full_download_20261030/download_manifest.json \
        --root /home/bingxing2/home/scx8178/energywam_storage \
        [--workers 2] [--only dataset_OpenDrawer] [--max-storage-gib 200]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat as statmod
import sys
import threading
import time
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import tarfile

SCHEMA = "energywam_robocasa365_download_v1"
CHUNK = 1 << 20
RESERVE_BYTES = 2 * (1 << 30)  # >= 2 GiB free-space reserve
ASSET_MARKER = ".robocasa365_asset.json"
LOCK_NAME = ".robocasa_download_worker.lock"
USER_AGENT = "energywam-robocasa-download-worker/1.0"

# Dataset already installed by hand: verify these counts before recording
# "existing" (info.json totals + on-disk file counts). See CLAUDE_REPORT.md.
KNOWN_EXISTING_CHECKS = {
    "dataset_OpenDrawer": {
        "total_episodes": 514,
        "total_frames": 137431,
        "videos": 1542,
        "parquet": 514,
    },
}


class StopRequested(Exception):
    """Graceful shutdown (SIGTERM/SIGINT) requested."""


class IntegrityError(Exception):
    """Downloaded/extracted content failed validation."""


class SafeExtractionError(Exception):
    """Archive member failed the safety checks."""


class BudgetError(Exception):
    """Space guard refused the operation."""


# --------------------------------------------------------------------------
# status persistence
# --------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


class StatusStore:
    """Per-item status JSON under ``<root>/status/<id>.json`` + aggregate."""

    def __init__(self, root: Path):
        self.dir = root / "status"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def item_path(self, item_id: str) -> Path:
        return self.dir / f"{item_id}.json"

    def load(self, item_id: str) -> dict:
        path = self.item_path(item_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return {"id": item_id, "state": "new"}

    def save(self, record: dict) -> None:
        record = dict(record)
        record["updated_at"] = _utcnow()
        with self._lock:
            atomic_write_json(self.item_path(record["id"]), record)

    def write_aggregate(self, records: list[dict], extra: dict | None = None) -> dict:
        counts: dict[str, int] = {}
        for rec in records:
            counts[rec.get("state", "unknown")] = counts.get(rec.get("state", "unknown"), 0) + 1
        agg = {
            "generated_at": _utcnow(),
            "counts": counts,
            "entries": [
                {
                    "id": r.get("id"),
                    "kind": r.get("kind"),
                    "state": r.get("state"),
                    "install_path": r.get("install_path"),
                    "error": r.get("error"),
                }
                for r in records
            ],
        }
        if extra:
            agg.update(extra)
        with self._lock:
            atomic_write_json(self.dir / "aggregate.json", agg)
        return agg


# --------------------------------------------------------------------------
# space guard
# --------------------------------------------------------------------------


class SpaceGuard:
    """Free-space reserve plus a ceiling on this worker's download+extract use."""

    def __init__(self, root: Path, max_gib: float, reserve_bytes: int = RESERVE_BYTES):
        self.root = root
        self.budget_bytes = int(max_gib * (1 << 30))
        self.reserve_bytes = max(reserve_bytes, RESERVE_BYTES)  # reserve >= 2 GiB
        self._lock = threading.Lock()
        self._extracted_bytes = 0  # validated installs made by this run

    def free_bytes(self) -> int:
        st = os.statvfs(self.root)
        return st.f_bavail * st.f_frsize

    def archive_bytes_on_disk(self) -> int:
        total = 0
        archives = self.root / "archives"
        if archives.is_dir():
            for p in archives.iterdir():
                if p.is_file():
                    total += p.stat().st_size
        return total

    def used_bytes(self, prior_extracted: int = 0) -> int:
        return self.archive_bytes_on_disk() + prior_extracted + self._extracted_bytes

    def check_download(self, archive_bytes: int, prior_extracted: int = 0) -> None:
        if archive_bytes is None or archive_bytes <= 0:
            raise BudgetError("unknown archive size; root must supply actual total")
        with self._lock:
            free = self.free_bytes()
            if free - archive_bytes < self.reserve_bytes:
                raise BudgetError(
                    f"free {free >> 30} GiB minus {archive_bytes >> 30} GiB archive "
                    f"would break the {self.reserve_bytes >> 30} GiB reserve"
                )
            used = self.used_bytes(prior_extracted)
            if used + archive_bytes > self.budget_bytes:
                raise BudgetError(
                    f"--max-storage-gib ceiling: used {used} + archive {archive_bytes} "
                    f"> budget {self.budget_bytes}"
                )

    def check_extract(self, prior_extracted: int = 0, expected_bytes: int = 0) -> None:
        with self._lock:
            free = self.free_bytes()
            if free - expected_bytes < self.reserve_bytes:
                raise BudgetError(
                    f"only {free >> 30} GiB free; refusing extraction below "
                    f"{self.reserve_bytes >> 30} GiB reserve"
                )
            if self.used_bytes(prior_extracted) + expected_bytes > self.budget_bytes:
                raise BudgetError("expected extraction exceeds the configured storage ceiling")

    def record_extracted(self, nbytes: int) -> None:
        with self._lock:
            self._extracted_bytes += nbytes

    def over_budget(self, prior_extracted: int = 0) -> bool:
        with self._lock:
            return self.used_bytes(prior_extracted) > self.budget_bytes


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


@dataclass
class Entry:
    id: str
    kind: str
    key: str
    url: str
    relative_destination: str
    archive_bytes: int | None
    range_supported: bool
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def archive_name(self) -> str:
        return f"{self.id}.{'tar' if self.kind == 'dataset' else 'zip'}"

    @property
    def dest_parts(self) -> tuple[str, ...]:
        return PurePosixPath(self.relative_destination).parts


def load_manifest(path: Path) -> list[Entry]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError(f"{path}: not a manifest with an 'entries' list")
    entries: list[Entry] = []
    seen: set[str] = set()
    for raw in data["entries"]:
        missing = [k for k in ("id", "kind", "key", "url", "relative_destination") if not raw.get(k)]
        if missing:
            raise ValueError(f"manifest entry missing {missing}: {raw}")
        if raw["kind"] not in ("dataset", "asset"):
            raise ValueError(f"manifest entry bad kind {raw['kind']!r}: {raw['id']}")
        if raw["id"] in seen:
            raise ValueError(f"duplicate manifest id {raw['id']}")
        seen.add(raw["id"])
        dest = PurePosixPath(raw["relative_destination"])
        if dest.is_absolute() or ".." in dest.parts:
            raise ValueError(f"unsafe relative_destination {raw['relative_destination']!r}")
        entries.append(
            Entry(
                id=raw["id"],
                kind=raw["kind"],
                key=raw["key"],
                url=raw["url"],
                relative_destination=raw["relative_destination"],
                archive_bytes=raw.get("archive_bytes"),
                range_supported=bool(raw.get("range_supported", False)),
                raw=raw,
            )
        )
    return entries


# --------------------------------------------------------------------------
# integrity helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def looks_like_html(head: bytes) -> bool:
    head = head.lstrip()[:64].lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or head.startswith(b"<head")


def validate_archive_file(path: Path, kind: str) -> None:
    """Reject HTML error pages and obviously wrong containers before install."""
    with open(path, "rb") as f:
        head = f.read(512)
    if not head:
        raise IntegrityError(f"{path.name}: empty archive")
    if looks_like_html(head):
        raise IntegrityError(f"{path.name}: HTML error page saved as archive")
    if kind == "dataset":
        if not tarfile.is_tarfile(path):
            raise IntegrityError(f"{path.name}: not a tar archive")
    else:
        if not zipfile.is_zipfile(path):
            raise IntegrityError(f"{path.name}: not a zip archive")


# --------------------------------------------------------------------------
# download (resumable .part files)
# --------------------------------------------------------------------------


_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


def _download_once(
    url: str,
    part_path: Path,
    expected_bytes: int | None,
    range_supported: bool,
    stop: threading.Event | None,
) -> tuple[int, int | None]:
    """One HTTP transfer attempt. Returns (size_on_disk, known_total)."""
    offset = part_path.stat().st_size if part_path.exists() else 0
    if expected_bytes is not None and offset > expected_bytes:
        part_path.unlink()
        offset = 0
    if offset and not range_supported:
        offset = 0
    if expected_bytes is not None and offset == expected_bytes:
        return offset, expected_bytes

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=120) as resp:
        status = resp.status
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise IntegrityError(f"HTTP {status} Content-Type {ctype!r} (HTML error page)")

        known_total: int | None = None
        if status == 206:
            cr = resp.headers.get("Content-Range", "")
            m = _CONTENT_RANGE.match(cr.strip())
            if not m:
                raise IntegrityError(f"unparsable Content-Range {cr!r}")
            start, end, total_s = int(m.group(1)), int(m.group(2)), m.group(3)
            if total_s != "*":
                known_total = int(total_s)
            if start != offset:
                raise IntegrityError(
                    f"Content-Range start {start} != requested offset {offset}"
                )
            if offset and end < start:
                raise IntegrityError(f"Content-Range end {end} < start {start}")
            if known_total is not None and end > known_total - 1:
                raise IntegrityError(f"Content-Range end {end} beyond total {known_total}")
        elif status == 200:
            if offset:
                # Server ignored the Range header: restart cleanly.
                offset = 0
            clen = resp.headers.get("Content-Length")
            if clen and clen.isdigit():
                known_total = int(clen)
        else:
            raise IntegrityError(f"unexpected HTTP status {status}")

        if expected_bytes is not None and known_total is not None and known_total != expected_bytes:
            raise IntegrityError(
                f"server total {known_total} != manifest archive_bytes {expected_bytes}"
            )

        mode = "ab" if offset else "wb"
        with open(part_path, mode) as out:
            first = True
            while True:
                if stop is not None and stop.is_set():
                    raise StopRequested()
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                if first:
                    first = False
                    if looks_like_html(chunk):
                        raise IntegrityError("response body is an HTML error page")
                out.write(chunk)
                if expected_bytes is not None and out.tell() > expected_bytes:
                    raise IntegrityError("response body exceeds manifest archive size")

    size = part_path.stat().st_size if part_path.exists() else 0
    if size == 0:
        raise IntegrityError("empty response body")
    return size, known_total


def download_archive(
    url: str,
    part_path: Path,
    expected_bytes: int | None,
    range_supported: bool,
    *,
    retries: int = 5,
    stop: threading.Event | None = None,
    log=print,
) -> int:
    """Download ``url`` into ``part_path`` (resumable). Returns final size."""
    attempt = 0
    while True:
        attempt += 1
        try:
            size, known_total = _download_once(url, part_path, expected_bytes, range_supported, stop)
            target = expected_bytes if expected_bytes is not None else known_total
            if target is not None and size < target:
                # partial range / dropped connection: resume on next attempt
                raise IntegrityError(f"truncated body: {size}/{target} bytes")
            if target is not None and size > target:
                part_path.unlink(missing_ok=True)
                raise IntegrityError(f"oversized body: {size}/{target} bytes")
            if expected_bytes is not None and size != expected_bytes:
                raise IntegrityError(
                    f"size {size} != manifest archive_bytes {expected_bytes}"
                )
            return size
        except StopRequested:
            raise
        except Exception as exc:
            if attempt >= retries:
                raise
            delay = min(60.0, 2.0 ** (attempt - 1))
            log(f"    retry {attempt}/{retries} in {delay:.0f}s: {type(exc).__name__}: {exc}")
            if stop is not None:
                if stop.wait(delay):
                    raise StopRequested()
            else:
                time.sleep(delay)


def record_sha256(final_path: Path, digest: str) -> None:
    (final_path.parent / f"{final_path.name}.sha256").write_text(
        f"{digest}  {final_path.name}\n"
    )


def finalize_archive(part_path: Path, final_path: Path, kind: str, expected_bytes: int | None) -> str:
    """Validate the .part file and promote it to the on-disk archive. Returns sha256."""
    size = part_path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise IntegrityError(f"size {size} != manifest archive_bytes {expected_bytes}")
    validate_archive_file(part_path, kind)
    digest = sha256_file(part_path)
    os.replace(part_path, final_path)
    record_sha256(final_path, digest)
    return digest


# --------------------------------------------------------------------------
# safe extraction
# --------------------------------------------------------------------------


def check_member_name(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise SafeExtractionError(f"bad member name {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise SafeExtractionError(f"absolute member name {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise SafeExtractionError(f"path traversal in member name {name!r}")
    return p


def _safe_target(staging: Path, rel: PurePosixPath) -> Path:
    dest = staging.joinpath(*rel.parts)
    resolved = dest.resolve()
    root = staging.resolve()
    if resolved != root and root not in resolved.parents:
        raise SafeExtractionError(f"member escapes staging dir: {rel}")
    return dest


def _copy_stream(src, dest: Path, stop: threading.Event | None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dest, "wb") as out:
        while True:
            if stop is not None and stop.is_set():
                raise StopRequested()
            chunk = src.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
    return written


def extract_dataset_tar(archive: Path, staging: Path, stop: threading.Event | None = None) -> int:
    """Extract only ``lerobot/`` members of the TAR into ``staging``.

    The harmless top-level ``README.md`` (and anything else outside
    ``lerobot/``) is ignored. Returns total extracted bytes.
    """
    staging.mkdir(parents=True, exist_ok=True)
    total = 0
    seen = 0
    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            path = check_member_name(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise SafeExtractionError(
                    f"unsupported tar member type for {member.name!r} "
                    f"(symlinks/hardlinks/devices are rejected)"
                )
            if not path.parts or path.parts[0] != "lerobot" or len(path.parts) < 2:
                continue  # top-level README.md and other non-lerobot entries
            rel = PurePosixPath(*path.parts[1:])
            dest = _safe_target(staging, rel)
            src = tf.extractfile(member)
            if src is None:
                raise SafeExtractionError(f"unreadable tar member {member.name!r}")
            total += _copy_stream(src, dest, stop)
            seen += 1
    if not seen:
        raise SafeExtractionError(f"{archive.name}: no lerobot/ members found")
    return total


def _is_harmless_zip_name(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (name == "README.md" or "__MACOSX" in parts
            or PurePosixPath(name).name in (".DS_Store", "Thumbs.db"))


def extract_asset_zip(
    archive: Path,
    staging: Path,
    expected_roots: set[str],
    stop: threading.Event | None = None,
) -> int:
    """Extract a RoboCasa asset ZIP into ``staging``.

    The ZIP's asset root (e.g. ``textures/``) must match one of
    ``expected_roots``; that root is stripped so ``staging`` holds the asset
    dir's contents. ``__MACOSX`` / ``.DS_Store`` are ignored; symlinks and
    traversal/absolute names are rejected. Returns total extracted bytes.
    """
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        entries = []
        for info in zf.infolist():
            name = info.filename
            if _is_harmless_zip_name(name):
                continue
            check_member_name(name)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == statmod.S_IFLNK:
                raise SafeExtractionError(f"symlink in zip: {name!r}")
            if info.is_dir():
                continue
            entries.append(info)
        if not entries:
            raise SafeExtractionError(f"{archive.name}: no usable members")

        roots = {PurePosixPath(e.filename).parts[0] for e in entries}
        if len(roots) != 1:
            raise SafeExtractionError(
                f"{archive.name}: expected a single zip root, found {sorted(roots)}"
            )

        # The asset root (possibly nested, e.g. robocasa/models/assets/textures)
        # must appear at a uniform index and match an expected basename.
        strip_at: int | None = None
        for info in entries:
            parts = PurePosixPath(info.filename).parts
            idx = next((i for i, part in enumerate(parts[:-1]) if part in expected_roots), None)
            if idx is None and parts[0] in expected_roots:
                idx = 0
            if idx is None:
                raise SafeExtractionError(
                    f"{archive.name}: zip root {sorted(roots)} does not match expected "
                    f"asset root {sorted(expected_roots)} in {info.filename!r}"
                )
            if strip_at is None:
                strip_at = idx
            elif strip_at != idx:
                raise SafeExtractionError(
                    f"{archive.name}: inconsistent asset root depth in {info.filename!r}"
                )

        assert strip_at is not None
        total = 0
        for info in entries:
            parts = PurePosixPath(info.filename).parts
            rel = PurePosixPath(*parts[strip_at + 1 :])
            if not rel.parts:
                continue
            dest = _safe_target(staging, rel)
            with zf.open(info) as src:
                total += _copy_stream(src, dest, stop)
    return total


# --------------------------------------------------------------------------
# dataset verification
# --------------------------------------------------------------------------


def read_info_json(dataset_dir: Path) -> dict:
    for candidate in (dataset_dir / "info.json", dataset_dir / "meta" / "info.json"):
        if candidate.is_file():
            try:
                info = json.loads(candidate.read_text())
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"{candidate}: invalid JSON ({exc})") from exc
            if not isinstance(info, dict):
                raise IntegrityError(f"{candidate}: not a JSON object")
            return info
    raise IntegrityError(f"{dataset_dir}: no info.json found")


def count_dataset_files(dataset_dir: Path) -> tuple[int, int]:
    data_dir = dataset_dir / "data"
    videos_dir = dataset_dir / "videos"
    parquet = sum(1 for p in (data_dir if data_dir.is_dir() else dataset_dir).rglob("*.parquet"))
    videos = sum(
        1
        for p in (videos_dir if videos_dir.is_dir() else dataset_dir).rglob("*")
        if p.suffix in (".mp4", ".webm", ".avi", ".mov")
    )
    return parquet, videos


def expected_counts_from_info(info: dict) -> dict:
    episodes = info.get("total_episodes")
    if not isinstance(episodes, int) or episodes <= 0:
        raise IntegrityError(f"info.json total_episodes invalid: {episodes!r}")
    out = {"total_episodes": episodes, "total_frames": info.get("total_frames")}
    if isinstance(info.get("total_parquet"), int):
        out["parquet"] = info["total_parquet"]
    else:
        # LeRobot v2 chunks_size groups episode files into directories; it
        # does not merge 1000 episodes into one parquet.
        out["parquet"] = episodes
    if isinstance(info.get("total_videos"), int):
        out["videos"] = info["total_videos"]
    return out


def verify_lerobot_dir(dataset_dir: Path, expected: dict | None = None) -> dict:
    """Verify a LeRobot dataset dir against its own ``info.json`` counts.

    ``expected`` (optional) adds hard requirements, e.g. the known OpenDrawer
    counts ``{"total_episodes": 514, "total_frames": 137431, "videos": 1542,
    "parquet": 514}``. Raises :class:`IntegrityError` on any mismatch; returns
    the measured counts on success.
    """
    if not dataset_dir.is_dir():
        raise IntegrityError(f"{dataset_dir}: not a directory")
    info = read_info_json(dataset_dir)
    want = expected_counts_from_info(info)
    if expected:
        for k in ("total_episodes", "total_frames"):
            if k in expected and info.get(k) != expected[k]:
                raise IntegrityError(
                    f"{dataset_dir.name}: info.json {k}={info.get(k)!r} != expected {expected[k]!r}"
                )
    parquet, videos = count_dataset_files(dataset_dir)
    if parquet != want["parquet"]:
        raise IntegrityError(
            f"{dataset_dir.name}: {parquet} parquet files != info.json {want['parquet']}"
        )
    if "videos" in want and videos != want["videos"]:
        raise IntegrityError(
            f"{dataset_dir.name}: {videos} videos != info.json {want['videos']}"
        )
    if "videos" not in want and videos < 1:
        raise IntegrityError(f"{dataset_dir.name}: no video files found")
    if expected:
        for k, actual in (("parquet", parquet), ("videos", videos)):
            if k in expected and actual != expected[k]:
                raise IntegrityError(
                    f"{dataset_dir.name}: {actual} {k} files != expected {expected[k]}"
                )
    return {
        "total_episodes": want["total_episodes"],
        "total_frames": info.get("total_frames"),
        "parquet": parquet,
        "videos": videos,
    }


# --------------------------------------------------------------------------
# install logic
# --------------------------------------------------------------------------


def _rmtree(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _is_absent_or_empty(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def expected_zip_roots(entry: Entry) -> set[str]:
    return {PurePosixPath(entry.relative_destination).parts[-1], entry.key}


def asset_marker_valid(dest: Path, entry: Entry) -> bool:
    marker = dest / ASSET_MARKER
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("id") == entry.id and data.get("key") == entry.key


def resolve_asset_target(root: Path, entry: Entry) -> tuple[Path, str]:
    """Pick the install dir for an asset: manifest destination is authoritative.

    Falls back to the independent ``<root>/assets/robocasa365/<key>`` tree when
    the manifest destination already holds non-empty data (e.g. repository
    provided files) that we must not overwrite. Returns (target, note).
    """
    manifest_dest = root.joinpath(*entry.dest_parts)
    if _is_absent_or_empty(manifest_dest):
        return manifest_dest, "manifest_destination"
    fallback = root / "assets" / "robocasa365" / entry.key
    if _is_absent_or_empty(fallback):
        return fallback, (
            "fallback_independent_path (manifest destination non-empty; not overwritten)"
        )
    raise IntegrityError(
        f"both {manifest_dest} and {fallback} are non-empty; refusing to overwrite"
    )


def install_dataset(
    root: Path,
    entry: Entry,
    archive: Path,
    guard: SpaceGuard,
    stop: threading.Event | None,
    prior_extracted: int,
) -> tuple[Path, dict]:
    dest = root.joinpath(*entry.dest_parts)
    if dest.exists():
        raise IntegrityError(f"{dest} already exists; refusing to overwrite user data")
    staging = dest.parent / f".{dest.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        guard.check_extract(prior_extracted, int(entry.raw.get("uncompressed_bytes", entry.archive_bytes) or 0))
        extracted = extract_dataset_tar(archive, staging, stop)
        counts = verify_lerobot_dir(staging, KNOWN_EXISTING_CHECKS.get(entry.id))
        counts["extracted_bytes"] = extracted
        if dest.exists():  # lost a race
            raise IntegrityError(f"{dest} appeared during extraction; not overwriting")
        staging.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, dest)
    except BaseException:
        _rmtree(staging)
        raise
    return dest, counts


def install_asset(
    root: Path,
    entry: Entry,
    archive: Path,
    guard: SpaceGuard,
    stop: threading.Event | None,
    prior_extracted: int,
) -> tuple[Path, dict]:
    target, note = resolve_asset_target(root, entry)
    staging = root / "assets" / "robocasa365" / f".staging-{entry.id}-{os.getpid()}"
    try:
        guard.check_extract(prior_extracted, int(entry.raw.get("uncompressed_bytes", entry.archive_bytes) or 0))
        extracted = extract_asset_zip(archive, staging, expected_zip_roots(entry), stop)
        files = sum(1 for p in staging.rglob("*") if p.is_file())
        if entry.raw.get("expected_files") is not None and files != entry.raw["expected_files"]:
            raise IntegrityError(f"asset file count {files} differs from registry inventory {entry.raw['expected_files']}")
        marker = {
            "id": entry.id,
            "key": entry.key,
            "relative_destination": entry.relative_destination,
            "installed_path": str(target),
            "note": note,
            "extracted_bytes": extracted,
            "files": files,
            "installed_at": _utcnow(),
        }
        atomic_write_json(staging / ASSET_MARKER, marker)
        if not _is_absent_or_empty(target):
            raise IntegrityError(f"{target} became non-empty during extraction")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        _rmtree(staging)
        raise
    return target, {"note": note, "extracted_bytes": extracted, "files": marker["files"]}


# --------------------------------------------------------------------------
# per-entry orchestration
# --------------------------------------------------------------------------


@dataclass
class Context:
    root: Path
    guard: SpaceGuard
    status: StatusStore
    stop: threading.Event
    retries: int = 5
    prior_extracted: int = 0


def process_entry(entry: Entry, ctx: Context) -> dict:
    rec = ctx.status.load(entry.id)
    rec.update(id=entry.id, kind=entry.kind, key=entry.key,
               relative_destination=entry.relative_destination)
    try:
        return _process_entry(entry, ctx, rec)
    except StopRequested:
        rec["state"] = "interrupted"
        ctx.status.save(rec)
        return rec
    except BudgetError as exc:
        rec.update(state="blocked_budget", error=str(exc))
        ctx.status.save(rec)
        return rec
    except OSError as exc:
        if exc.errno in (28, 122):  # ENOSPC / EDQUOT: stop the whole queue.
            ctx.stop.set()
            rec.update(state="blocked_budget", error=f"storage quota/full: {exc}")
        else:
            rec.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        ctx.status.save(rec)
        return rec
    except Exception as exc:  # recorded, independent items keep going
        rec.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        ctx.status.save(rec)
        return rec


def _process_entry(entry: Entry, ctx: Context, rec: dict) -> dict:
    dest = ctx.root.joinpath(*entry.dest_parts)

    # --- already installed? validate properly, never trust bare existence ---
    if entry.kind == "dataset":
        if dest.is_dir():
            try:
                counts = verify_lerobot_dir(dest, KNOWN_EXISTING_CHECKS.get(entry.id))
            except IntegrityError as exc:
                raise IntegrityError(
                    f"destination exists but failed validation ({exc}); "
                    f"not overwriting -- manual cleanup required"
                ) from exc
            rec.update(state="existing", install_path=str(dest), counts=counts,
                       error=None, source="validated_existing_dataset")
            ctx.status.save(rec)
            return rec
    else:
        if dest.is_dir() and asset_marker_valid(dest, entry):
            rec.update(state="existing", install_path=str(dest), error=None,
                       source="asset_marker")
            ctx.status.save(rec)
            return rec

    # --- archive on disk already? (resume source named by id) ---
    archive = ctx.root / "archives" / entry.archive_name
    part = archive.with_name(archive.name + ".part")

    if entry.archive_bytes is None:
        raise BudgetError(
            f"unknown archive size for {entry.id}; root must supply actual total"
        )

    digest = None
    if archive.is_file() and archive.stat().st_size == entry.archive_bytes:
        try:
            validate_archive_file(archive, entry.kind)
            digest = sha256_file(archive)
        except Exception:
            _rmtree(archive)
    elif archive.is_file():
        _rmtree(archive)
    if archive.is_file() and digest:
        record_sha256(archive, digest)

    if not archive.is_file():
        ctx.guard.check_download(entry.archive_bytes, ctx.prior_extracted)
        rec.update(state="downloading", error=None)
        ctx.status.save(rec)
        download_archive(
            entry.url,
            part,
            entry.archive_bytes,
            entry.range_supported,
            retries=ctx.retries,
            stop=ctx.stop,
            log=lambda msg: print(f"  [{entry.id}] {msg}", flush=True),
        )
        rec["state"] = "verifying_archive"
        ctx.status.save(rec)
        digest = finalize_archive(part, archive, entry.kind, entry.archive_bytes)
    rec["archive_sha256"] = digest

    # --- extract + validate + atomic install ---
    rec["state"] = "extracting"
    ctx.status.save(rec)
    if entry.kind == "dataset":
        install_path, counts = install_dataset(
            ctx.root, entry, archive, ctx.guard, ctx.stop, ctx.prior_extracted
        )
        rec["counts"] = counts
        rec["extracted_bytes"] = counts.get("extracted_bytes", 0)
    else:
        install_path, meta = install_asset(
            ctx.root, entry, archive, ctx.guard, ctx.stop, ctx.prior_extracted
        )
        rec["install_note"] = meta["note"]
        rec["extracted_bytes"] = meta["extracted_bytes"]

    ctx.guard.record_extracted(rec["extracted_bytes"])
    # --- validated: cleanup our archive (keep sha256 record + unrelated files) ---
    _rmtree(part)
    _rmtree(archive)
    rec.update(state="done", install_path=str(install_path), error=None, cleaned_archive=True)
    ctx.status.save(rec)
    print(f"  [{entry.id}] done -> {install_path}", flush=True)
    return rec


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class _Lock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise SystemExit(
                f"another worker already holds {self.path}; refusing to run twice"
            )
        os.truncate(self.fd, 0)
        os.write(self.fd, f"{os.getpid()}\n".encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--only", action="append", default=None,
                   help="process only this entry id (repeatable)")
    p.add_argument("--max-storage-gib", type=float, default=200.0)
    p.add_argument("--retries", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    (root / "archives").mkdir(exist_ok=True)

    entries = load_manifest(args.manifest)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {e.id for e in entries}
        if unknown:
            print(f"unknown --only ids: {sorted(unknown)}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e.id in wanted]

    stop = threading.Event()

    def _handle(signum, frame):
        print(f"\nreceived signal {signum}; finishing current writes, preserving partials",
              flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    status = StatusStore(root)
    guard = SpaceGuard(root, args.max_storage_gib)
    prior_extracted = sum(
        (status.load(e.id).get("extracted_bytes") or 0) for e in entries
    )
    ctx = Context(root=root, guard=guard, status=status, stop=stop,
                  retries=args.retries, prior_extracted=prior_extracted)

    known_total = sum((e.archive_bytes or 0) for e in entries)
    unknown = [e.id for e in entries if e.archive_bytes is None]
    print(f"{len(entries)} entries selected; known archive total "
          f"{known_total / (1 << 30):.1f} GiB; budget {args.max_storage_gib:.0f} GiB; "
          f"free {guard.free_bytes() >> 30} GiB", flush=True)
    if unknown:
        print(f"unknown archive sizes (will report, not download): {unknown}", flush=True)

    with _Lock(root / LOCK_NAME):
        results: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(process_entry, e, ctx): e for e in entries}
            for fut in concurrent.futures.as_completed(futures):
                if fut.cancelled():
                    continue
                results.append(fut.result())
                status.write_aggregate(
                    results, extra={"interrupted": stop.is_set(), "root": str(root)}
                )
                if stop.is_set():
                    for f in futures:
                        f.cancel()

        agg = status.write_aggregate(results, extra={"interrupted": stop.is_set(),
                                                     "root": str(root)})
        done = sum(1 for r in results if r.get("state") in ("done", "existing"))
        failed = [r for r in results if r.get("state") not in ("done", "existing", "interrupted")]
        print(json.dumps({"counts": agg["counts"], "failed": [r["id"] for r in failed]},
                         indent=2), flush=True)
        for r in failed:
            print(f"FAILED {r['id']}: {r.get('error')}", flush=True)
        if guard.over_budget(prior_extracted):
            print("WARNING: --max-storage-gib ceiling reached; further items blocked",
                  flush=True)
        if stop.is_set():
            return 143
        return 0 if done == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
