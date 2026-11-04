from pathlib import Path
import tempfile
import unittest
from package_tp2_evidence import build_archive, files_in, safe_name, verify_archive


class CustodyTests(unittest.TestCase):
    def test_unsafe_member_names_rejected(self):
        for name in ("", "/absolute", "../outside", "data/../../outside"):
            with self.assertRaises(ValueError):
                safe_name(name)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "link").symlink_to("/etc/hostname")
            with self.assertRaises(ValueError):
                files_in(root, "data")

    def test_streamed_archive_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            (source / "a").write_bytes(b"raw evidence\x00\xff")
            (source / "b").write_text("synthetic test only")
            archive = root / "test.tar.zst"
            result = build_archive(archive, files_in(source, "data"))
            self.assertEqual(set(result["members"]), {"data/a", "data/b"})
            verify_archive(archive, result["members"])
            with self.assertRaises(FileExistsError):
                build_archive(archive, files_in(source, "data"))

    def test_duplicate_names_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a"
            source.write_bytes(b"x")
            with self.assertRaises(ValueError):
                build_archive(root / "bad.tar.zst", [(source, "a"), (source, "a")])


if __name__ == "__main__":
    unittest.main()
