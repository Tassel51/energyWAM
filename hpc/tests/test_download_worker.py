"""Tests for ``hpc/scripts/robocasa_download_worker.py``.

Stdlib-only (unittest). Run::

    python3 hpc/tests/test_download_worker.py

Covers: extraction security (traversal/absolute/symlink/zip-root), resumable
download validation (strict Content-Range / size / HTML-error rejection) and
the existing-dataset validation path on a small LeRobot fixture (no network is
used except loopback HTTP fixtures).
"""

from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import robocasa_download_worker as w  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_lerobot_dir(
    path: Path,
    *,
    episodes: int = 2,
    frames: int = 10,
    videos: int | None = None,
    parquet: int | None = None,
    total_videos_field: bool = True,
    total_parquet_field: bool = True,
) -> dict:
    """Small LeRobot dataset tree with deterministic counts."""
    videos = episodes if videos is None else videos
    parquet = episodes if parquet is None else parquet
    info = {"total_episodes": episodes, "total_frames": frames}
    if total_videos_field:
        info["total_videos"] = videos
    if total_parquet_field:
        info["total_parquet"] = parquet
    path.mkdir(parents=True, exist_ok=True)
    (path / "info.json").write_text(json.dumps(info))
    for i in range(parquet):
        p = path / "data" / "chunk-0" / f"file-{i}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PARQUET" + bytes([i % 256]))
    for i in range(videos):
        p = path / "videos" / "chunk-0" / "cam" / f"file-{i}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"MP4" + bytes([i % 256]))
    return info


def make_dataset_tar(path: Path, *, extra_top_level: bool = True) -> int:
    """TAR with harmless top-level README.md plus lerobot/. Returns byte size."""
    info = {"total_episodes": 2, "total_frames": 10, "total_videos": 2, "total_parquet": 2}
    members = {
        "README.md": b"robocasa365 export\n",
        "lerobot/info.json": json.dumps(info).encode(),
        "lerobot/data/chunk-0/file-0.parquet": b"PARQUET0",
        "lerobot/data/chunk-0/file-1.parquet": b"PARQUET1",
        "lerobot/videos/chunk-0/cam/file-0.mp4": b"MP40",
        "lerobot/videos/chunk-0/cam/file-1.mp4": b"MP41",
    }
    if not extra_top_level:
        members.pop("README.md")
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path.stat().st_size


def make_asset_zip(path: Path, root: str = "textures", *, mac_junk: bool = True) -> int:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{root}/a/tex.png", b"PNGDATA")
        zf.writestr(f"{root}/a/more.png", b"PNGDATA2")
        if mac_junk:
            zf.writestr(f"__MACOSX/{root}/._tex.png", b"junk")
            zf.writestr(f"{root}/.DS_Store", b"junk")
    return path.stat().st_size


def make_entry(**kw) -> w.Entry:
    base = dict(
        id="dataset_Test",
        kind="dataset",
        key="target/atomic/Test/20250714/lerobot.tar",
        url="http://127.0.0.1:9/unused",
        relative_destination="datasets/robocasa365/v1.0/target/atomic/Test/20250714/lerobot",
        archive_bytes=10,
        range_supported=False,
    )
    base.update(kw)
    return w.Entry(**base)


def make_ctx(root: Path, max_gib: float = 200.0) -> w.Context:
    return w.Context(
        root=root,
        guard=w.SpaceGuard(root, max_gib),
        status=w.StatusStore(root),
        stop=threading.Event(),
        retries=1,
    )


# --------------------------------------------------------------------------
# loopback HTTP fixture
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        srv = self.server
        payload = srv.payload
        mode = srv.mode
        rng = self.headers.get("Range")
        if mode == "html":
            body = b"<html><body>404 not found</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        start = 0
        m = re.match(r"bytes=(\d+)-$", rng or "")
        if m:
            start = int(m.group(1))
        if rng and mode == "range":
            if start >= len(payload):
                self.send_response(416)
                self.end_headers()
                return
            body = payload[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if rng and mode == "bad-range":
            body = payload[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start + 7}-{len(payload) - 1}/{len(payload)}"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # "no-range" and "ignore-range": always answer 200 with the full body
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_server(payload: bytes, mode: str):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.payload = payload
    server.mode = mode
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/file"


# --------------------------------------------------------------------------
# extraction security
# --------------------------------------------------------------------------


class TestExtractionSecurity(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="rdw-test-"))
        self.staging = self.tmp / "staging"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tar_with(self, members: dict[str, object]) -> Path:
        archive = self.tmp / "evil.tar"
        with tarfile.open(archive, "w") as tf:
            for name, payload in members.items():
                if isinstance(payload, tuple):  # (kind, target) symlink/hardlink
                    kind, target = payload
                    ti = tarfile.TarInfo(name)
                    if kind == "sym":
                        ti.type = tarfile.SYMTYPE
                    else:
                        ti.type = tarfile.LNKTYPE
                    ti.linkname = target
                    tf.addfile(ti)
                else:
                    ti = tarfile.TarInfo(name)
                    ti.size = len(payload)
                    tf.addfile(ti, io.BytesIO(payload))
        return archive

    def test_tar_rejects_traversal_absolute_and_links(self):
        cases = {
            "traversal": {"lerobot/../evil.txt": b"x"},
            "absolute": {"/etc/passwd": b"x"},
            "symlink": {"lerobot/link": ("sym", "/etc/passwd")},
            "hardlink": {"lerobot/link": ("hard", "lerobot/info.json")},
        }
        for label, members in cases.items():
            with self.subTest(label):
                archive = self._tar_with(members)
                with self.assertRaises(w.SafeExtractionError):
                    w.extract_dataset_tar(archive, self.staging)
                self.assertFalse((self.tmp / "evil.txt").exists())
                self.assertFalse(Path("/etc/passwd.drw").exists())

    def test_tar_extracts_only_lerobot(self):
        archive = self.tmp / "ds.tar"
        make_dataset_tar(archive)
        total = w.extract_dataset_tar(archive, self.staging)
        self.assertGreater(total, 0)
        self.assertTrue((self.staging / "info.json").is_file())
        self.assertTrue((self.staging / "data/chunk-0/file-0.parquet").is_file())
        self.assertTrue((self.staging / "videos/chunk-0/cam/file-1.mp4").is_file())
        self.assertFalse((self.staging / "README.md").exists())  # top-level ignored

    def test_tar_without_lerobot_is_rejected(self):
        archive = self._tar_with({"README.md": b"hello"})
        with self.assertRaises(w.SafeExtractionError):
            w.extract_dataset_tar(archive, self.staging)

    def test_zip_rejects_traversal_absolute_and_symlink(self):
        archive = self.tmp / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("textures/../../evil.txt", b"x")
        with self.assertRaises(w.SafeExtractionError):
            w.extract_asset_zip(archive, self.staging, {"textures"})

        archive = self.tmp / "evil2.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("/abs/evil.txt", b"x")
        with self.assertRaises(w.SafeExtractionError):
            w.extract_asset_zip(archive, self.staging, {"textures"})

        archive = self.tmp / "link.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zi = zipfile.ZipInfo("textures/link")
            zi.external_attr = (0o120777 << 16)  # symlink
            zf.writestr(zi, "/etc/passwd")
        with self.assertRaises(w.SafeExtractionError):
            w.extract_asset_zip(archive, self.staging, {"textures"})

    def test_zip_root_must_match_expected_basename(self):
        archive = self.tmp / "badroot.zip"
        make_asset_zip(archive, root="evil_root")
        with self.assertRaises(w.SafeExtractionError):
            w.extract_asset_zip(archive, self.staging, {"textures"})

    def test_zip_good_root_strips_root_and_ignores_junk(self):
        archive = self.tmp / "tex.zip"
        make_asset_zip(archive, root="textures")
        total = w.extract_asset_zip(archive, self.staging, {"textures", "tex_key"})
        self.assertGreater(total, 0)
        self.assertTrue((self.staging / "a/tex.png").is_file())  # root stripped
        self.assertFalse((self.staging / "textures").exists())
        self.assertFalse((self.staging / "__MACOSX").exists())
        self.assertFalse((self.staging / "a/.DS_Store").exists())

    def test_zip_nested_root_matched_by_expected_basename(self):
        archive = self.tmp / "nested.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("robocasa/models/assets/textures/x.png", b"P")
        w.extract_asset_zip(archive, self.staging, {"textures"})
        self.assertTrue((self.staging / "x.png").is_file())

    def test_check_member_name(self):
        for bad in ("../x", "/x", "C:/x", "a\\b", "a/../../b", ""):
            with self.subTest(bad):
                with self.assertRaises(w.SafeExtractionError):
                    w.check_member_name(bad)
        self.assertEqual(w.check_member_name("lerobot/data/f.parquet").parts[0], "lerobot")


# --------------------------------------------------------------------------
# dataset verification
# --------------------------------------------------------------------------


class TestVerifyLerobotDir(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="rdw-test-"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fixture_verifies(self):
        ds = self.tmp / "lerobot"
        make_lerobot_dir(ds, episodes=2, frames=10, videos=2, parquet=2)
        counts = w.verify_lerobot_dir(ds)
        self.assertEqual(counts["total_episodes"], 2)
        self.assertEqual(counts["parquet"], 2)
        self.assertEqual(counts["videos"], 2)

    def test_count_mismatch_rejected(self):
        ds = self.tmp / "lerobot"
        make_lerobot_dir(ds, episodes=2, videos=2, parquet=2)
        (ds / "data/chunk-0/file-1.parquet").unlink()  # now 1 parquet, info says 2
        with self.assertRaises(w.IntegrityError):
            w.verify_lerobot_dir(ds)

    def test_known_expected_counts_enforced(self):
        ds = self.tmp / "lerobot"
        make_lerobot_dir(ds, episodes=2, frames=10, videos=2, parquet=2)
        w.verify_lerobot_dir(
            ds, {"total_episodes": 2, "total_frames": 10, "videos": 2, "parquet": 2}
        )
        with self.assertRaises(w.IntegrityError):
            w.verify_lerobot_dir(ds, {"parquet": 3})
        with self.assertRaises(w.IntegrityError):
            w.verify_lerobot_dir(ds, {"total_frames": 999})

    def test_missing_info_rejected(self):
        ds = self.tmp / "lerobot"
        ds.mkdir()
        with self.assertRaises(w.IntegrityError):
            w.verify_lerobot_dir(ds)

    def test_opendrawer_known_check_constant_shape(self):
        expected = w.KNOWN_EXISTING_CHECKS["dataset_OpenDrawer"]
        self.assertEqual(expected, {"total_episodes": 514, "total_frames": 137431,
                                    "videos": 1542, "parquet": 514})


# --------------------------------------------------------------------------
# resumable download validation
# --------------------------------------------------------------------------


class TestResumeDownload(unittest.TestCase):
    PAYLOAD = bytes(range(256)) * 64  # 16 KiB

    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="rdw-test-"))
        self.servers = []

    def tearDown(self):
        import shutil

        for srv in self.servers:
            srv.shutdown()
            srv.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _server(self, mode: str, payload: bytes | None = None):
        srv, url = start_server(self.PAYLOAD if payload is None else payload, mode)
        self.servers.append(srv)
        return url

    def test_resume_appends_with_valid_content_range(self):
        url = self._server("range")
        part = self.tmp / "a.tar.part"
        part.write_bytes(self.PAYLOAD[:100])
        size = w.download_archive(url, part, len(self.PAYLOAD), True, retries=1)
        self.assertEqual(size, len(self.PAYLOAD))
        self.assertEqual(part.read_bytes(), self.PAYLOAD)

    def test_no_range_support_restarts_from_zero(self):
        url = self._server("no-range")
        part = self.tmp / "a.tar.part"
        part.write_bytes(b"STALE-PARTIAL-DATA")
        w.download_archive(url, part, len(self.PAYLOAD), False, retries=1)
        self.assertEqual(part.read_bytes(), self.PAYLOAD)

    def test_server_ignoring_range_restarts_cleanly(self):
        url = self._server("ignore-range")
        part = self.tmp / "a.tar.part"
        part.write_bytes(self.PAYLOAD[:100])
        w.download_archive(url, part, len(self.PAYLOAD), True, retries=1)
        self.assertEqual(part.read_bytes(), self.PAYLOAD)

    def test_bad_content_range_rejected(self):
        url = self._server("bad-range")
        part = self.tmp / "a.tar.part"
        part.write_bytes(self.PAYLOAD[:100])
        with self.assertRaises(w.IntegrityError):
            w.download_archive(url, part, len(self.PAYLOAD), True, retries=1)

    def test_html_error_page_never_saved_as_archive(self):
        url = self._server("html")
        part = self.tmp / "a.tar.part"
        with self.assertRaises(w.IntegrityError):
            w.download_archive(url, part, len(self.PAYLOAD), False, retries=1)
        final = self.tmp / "a.tar"
        self.assertFalse(final.exists())

    def test_size_mismatch_with_manifest_rejected(self):
        url = self._server("range")
        part = self.tmp / "a.tar.part"
        with self.assertRaises(w.IntegrityError):
            w.download_archive(url, part, len(self.PAYLOAD) + 5, True, retries=1)

    def test_finalize_archive_checks_html_and_records_sha256(self):
        bad = self.tmp / "bad.tar"
        bad.write_text("<html>oops</html>")
        with self.assertRaises(w.IntegrityError):
            w.finalize_archive(bad, self.tmp / "out.tar", "dataset", bad.stat().st_size)

        archive = self.tmp / "ds.tar"
        expected = make_dataset_tar(archive)
        part = archive.with_name("ds.tar.part")
        part.write_bytes(archive.read_bytes())
        archive.unlink()
        digest = w.finalize_archive(part, archive, "dataset", expected)
        self.assertTrue(archive.is_file())
        sidecar = self.tmp / "ds.tar.sha256"
        self.assertTrue(sidecar.is_file())
        self.assertIn(digest, sidecar.read_text())


# --------------------------------------------------------------------------
# worker orchestration: existing dataset / install / skip
# --------------------------------------------------------------------------


class TestProcessEntry(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="rdw-test-"))
        self.root = self.tmp / "storage"
        self.root.mkdir()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset_entry(self, **kw) -> w.Entry:
        return make_entry(**kw)

    def test_existing_valid_dataset_recorded_and_skipped(self):
        rel = "datasets/robocasa365/v1.0/target/atomic/Test/20250714/lerobot"
        dest = self.root / rel
        make_lerobot_dir(dest, episodes=2, frames=10, videos=2, parquet=2)
        entry = self._dataset_entry(relative_destination=rel,
                                    url="http://127.0.0.1:9/never-fetched")
        rec = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec["state"], "existing")
        self.assertEqual(rec["source"], "validated_existing_dataset")
        self.assertEqual(rec["counts"]["total_episodes"], 2)
        self.assertFalse((self.root / "archives").exists())  # nothing downloaded

    def test_existing_invalid_dataset_never_overwritten(self):
        rel = "datasets/robocasa365/v1.0/target/atomic/Test/20250714/lerobot"
        dest = self.root / rel
        make_lerobot_dir(dest, episodes=2, videos=2, parquet=2)
        (dest / "data/chunk-0/file-1.parquet").unlink()  # broken counts
        entry = self._dataset_entry(relative_destination=rel)
        rec = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec["state"], "failed")
        self.assertIn("not overwriting", rec["error"])
        self.assertTrue((dest / "info.json").is_file())  # untouched

    def test_opendrawer_counts_verified_before_skip(self):
        """The known OpenDrawer counts gate the existing/skip decision."""
        rel = "datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250714/lerobot"
        dest = self.root / rel
        make_lerobot_dir(dest, episodes=514, frames=137431, videos=1542, parquet=514)
        entry = self._dataset_entry(id="dataset_OpenDrawer", relative_destination=rel,
                                    key="target/atomic/OpenDrawer/20250714/lerobot.tar")
        rec = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec["state"], "existing")
        self.assertEqual(rec["counts"]["total_frames"], 137431)

    def test_dataset_install_from_local_archive(self):
        rel = "datasets/robocasa365/v1.0/target/atomic/Test/20250714/lerobot"
        archive = self.root / "archives" / "dataset_Test.tar"
        archive.parent.mkdir(parents=True)
        size = make_dataset_tar(archive)
        entry = self._dataset_entry(relative_destination=rel, archive_bytes=size)
        rec = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec["state"], "done")
        dest = self.root / rel
        self.assertTrue((dest / "info.json").is_file())
        self.assertFalse((dest / "README.md").exists())
        self.assertFalse(archive.exists())  # own archive cleaned after validation
        self.assertTrue((self.root / "archives" / "dataset_Test.tar.sha256").is_file())
        self.assertEqual(rec["counts"]["parquet"], 2)

    def test_asset_install_marker_and_skip(self):
        rel = "sim/robocasa365/robocasa/models/assets/textures"
        archive = self.root / "archives" / "asset_textures.zip"
        archive.parent.mkdir(parents=True)
        size = make_asset_zip(archive, root="textures")
        entry = make_entry(
            id="asset_textures",
            kind="asset",
            key="textures",
            relative_destination=rel,
            archive_bytes=size,
            url="http://127.0.0.1:9/never-fetched",
        )
        rec = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec["state"], "done")
        dest = self.root / rel
        self.assertTrue((dest / "a/tex.png").is_file())
        self.assertTrue((dest / w.ASSET_MARKER).is_file())
        self.assertEqual(rec["install_note"], "manifest_destination")
        # second run: marker short-circuits, never touches the network
        rec2 = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec2["state"], "existing")
        self.assertEqual(rec2["source"], "asset_marker")

    def test_asset_falls_back_to_independent_path_when_target_occupied(self):
        rel = "sim/robocasa365/robocasa/models/assets/textures"
        occupied = self.root / rel
        occupied.mkdir(parents=True)
        (occupied / "repo_provided.png").write_bytes(b"keepme")
        archive = self.root / "archives" / "asset_textures.zip"
        archive.parent.mkdir(parents=True)
        size = make_asset_zip(archive, root="textures")
        entry = make_entry(id="asset_textures", kind="asset", key="textures",
                           relative_destination=rel, archive_bytes=size)
        rec = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec["state"], "done")
        fallback = self.root / "assets" / "robocasa365" / "textures"
        self.assertEqual(Path(rec["install_path"]), fallback)
        self.assertIn("fallback", rec["install_note"])
        self.assertTrue((fallback / "a/tex.png").is_file())
        self.assertEqual((occupied / "repo_provided.png").read_bytes(), b"keepme")

    def test_unknown_archive_size_reported_not_downloaded(self):
        entry = self._dataset_entry(archive_bytes=None)
        rec = w.process_entry(entry, make_ctx(self.root))
        self.assertEqual(rec["state"], "blocked_budget")
        self.assertIn("actual total", rec["error"])
        self.assertFalse((self.root / "archives").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
