import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import read_quality_release_v1 as probe


class PublicArchiveTests(unittest.TestCase):
    def make_fixture(self, directory, fault=None):
        payload = b"public-generated-fixture"
        name = "evidence/data.bin"
        manifest = {"original_member_count": 1, "embedded_manifest_member": "evidence/manifest.json",
                    "members": [{"member": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
        raw = json.dumps(manifest).encode()
        mpath = directory / "manifest.json"
        mpath.write_bytes(raw)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for member_name, data in [(name, payload), (manifest["embedded_manifest_member"], raw)]:
                if fault == "missing" and member_name == name:
                    continue
                if fault == "embedded" and member_name != name:
                    data = raw + b" "
                info = tarfile.TarInfo(member_name)
                info.size = len(data)
                if fault == "link" and member_name == name:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "/outside"
                    info.size = 0
                archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
                if fault == "duplicate" and member_name == name:
                    archive.addfile(info, io.BytesIO(data))
        apath = directory / "evidence.tar.gz"
        apath.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
        return apath, mpath, probe.identity(apath)["sha256"]

    def test_public_manifest_all_members_and_embedded_copy(self):
        with tempfile.TemporaryDirectory(dir=probe.HERE) as directory:
            result = probe.verify_archive(*self.make_fixture(Path(directory)), 1)
            self.assertTrue(result["passed"])
            self.assertEqual(result["total_members"], 2)
            self.assertFalse(result["downloaded_code_executed"])

    def test_missing_duplicate_links_and_wrong_embedded_copy_rejected(self):
        for fault in ("missing", "duplicate", "link", "embedded"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir=probe.HERE) as directory:
                with self.assertRaises(ValueError):
                    probe.verify_archive(*self.make_fixture(Path(directory), fault), 1)

    def test_noncanonical_paths_rejected(self):
        for path in ("/absolute", "../outside", "a/../outside", "a//b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                probe.safe_member(path)

    def test_final_curl_headers_after_connect_and_redirect(self):
        headers = probe.final_headers("HTTP/1.1 200 Connection established\r\n\r\n"
                                      "HTTP/2 302\r\nlocation: https://example.test\r\n\r\n"
                                      "HTTP/1.1 200 OK\r\ncontent-length: 5\r\n\r\n")
        self.assertEqual(headers["Content-Length"], "5")


if __name__ == "__main__":
    unittest.main()
