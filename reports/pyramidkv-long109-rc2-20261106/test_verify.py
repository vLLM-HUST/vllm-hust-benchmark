"""Mutation tests for archive extraction and semantic acceptance; standard library only."""

import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
import verify

REPORT = Path(__file__).resolve().parent


class Mutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pyramidkv-mutations-")
        cls.root = Path(cls.temp.name) / "raw"
        cls.root.mkdir()
        verify.extract(REPORT, cls.root)
        cls.source_tree = verify.source_identity(cls.root)
        cls.audit = verify.module(
            "mutation_audit",
            cls.root / "supplemental/source/scripts/model-comparison/audit.py",
        )
        cls.log = (cls.root / "runs/qwen35-on-558/server.log").read_text()
        cls.ids = [
            "long-readiness",
            *[r["id"] for r in verify.read(cls.root / "qwen35-requests.json")["long"]],
        ]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_valid_transactions(self):
        self.assertEqual(self.audit.transactions(self.log, "qwen35", self.ids), 129)

    def test_missing_long109(self):
        log = "\n".join(
            line
            for line in self.log.splitlines()
            if "scheduler commit request_id=cmpl-long-109-" not in line
        )
        with self.assertRaisesRegex(ValueError, "long-109"):
            self.audit.transactions(log, "qwen35", self.ids)

    def test_missing_rank_ack(self):
        log = "\n".join(
            line
            for line in self.log.splitlines()
            if not (
                "worker commit acknowledged request_id=cmpl-long-109-" in line
                and "worker_rank=1" in line
            )
        )
        with self.assertRaisesRegex(ValueError, "acknowledgement"):
            self.audit.transactions(log, "qwen35", self.ids)

    def test_numeric_suffix_rejected(self):
        for before, after in [
            ("worker_rank=1", "worker_rank=1junk"),
            ("destination_blocks=1 released", "destination_blocks=1junk released"),
        ]:
            with (
                self.subTest(after=after),
                self.assertRaisesRegex(ValueError, "Unparseable"),
            ):
                self.audit.transactions(
                    self.log.replace(before, after, 1), "qwen35", self.ids
                )

    def test_unknown_and_duplicate_commit(self):
        line = next(
            line
            for line in self.log.splitlines()
            if "scheduler commit request_id=" in line
        )
        for log in [
            self.log + "\n" + line,
            self.log.replace("cmpl-long-109-0-", "cmpl-long-unknown-0-"),
        ]:
            with self.assertRaises(ValueError):
                self.audit.transactions(log, "qwen35", self.ids)

    def test_bad_geometry(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.audit.transactions(
                self.log.replace("physical_tokens=2048", "physical_tokens=4096", 1),
                "qwen35",
                self.ids,
            )

    def test_final_activity(self):
        metrics = (self.root / "runs/qwen35-on-558/final-metrics.prom").read_text()
        import re

        for bad in [
            re.sub(r"(vllm:num_requests_running\{[^\n]*\}) 0.0", r"\1 1.0", metrics),
            re.sub(
                r"^vllm:num_requests_waiting\{[^\n]*\}.*\n", "", metrics, flags=re.M
            ),
        ]:
            self.assertNotEqual(metrics, bad)
            with self.assertRaises(ValueError):
                self.audit.idle(bad)

    def test_semantic_timing_mutation(self):
        path = self.root / "runs/qwen35-on-558/evaluation/long/long-109.result.json"
        original = path.read_bytes()
        try:
            value = json.loads(original)
            value["tpot_seconds"] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(
                ValueError, "Recomputed stream metrics mismatch"
            ):
                verify.validate(self.root)
        finally:
            path.write_bytes(original)

    def test_source_blob_identity(self):
        path = (
            self.root
            / "supplemental/source/src/vllm_ascend_pyramidkv/adapter/scheduler.py"
        )
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"# mutation\n")
            with self.assertRaisesRegex(ValueError, "Committed source blob mismatch"):
                verify.source_identity(self.root)
        finally:
            path.write_bytes(original)

    def test_startup_pool_mismatch(self):
        path = self.root / "runs/qwen35-on-558/startup-metrics.prom"
        original = path.read_bytes()
        try:
            altered = original.replace(b'num_gpu_blocks="558"', b'num_gpu_blocks="557"')
            self.assertNotEqual(altered, original)
            path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, "Native pool mismatch"):
                verify.validate(self.root)
        finally:
            path.write_bytes(original)

    def test_raw_sse_mismatch(self):
        import subprocess

        path = self.root / "runs/qwen35-on-558/evaluation/long/long-109.sse"
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b"[DONE]", b"[BROKEN]"))
            with self.assertRaises(subprocess.CalledProcessError):
                verify.validate(self.root)
        finally:
            path.write_bytes(original)

    def test_admission_after_first_completion(self):
        import subprocess

        path = self.root / "runs/qwen35-on-558/evaluation/telemetry.jsonl"
        original = path.read_bytes()
        rows = [json.loads(line) for line in original.splitlines()]
        results = verify.read(
            self.root / "runs/qwen35-on-558/evaluation/long/group.json"
        )["results"]
        first_end = min(row["end"] for row in results)
        try:
            changed = 0
            for row in rows:
                activity = sum(
                    value
                    for key, value in row["metrics"].items()
                    if key.startswith(
                        ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
                    )
                )
                if row["stage"] == "long" and activity == 128:
                    row["t"] = first_end + 1
                    changed += 1
            self.assertGreater(changed, 0)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(subprocess.CalledProcessError):
                verify.validate(self.root)
        finally:
            path.write_bytes(original)

    def test_delivery_tree_mutation(self):
        with tempfile.TemporaryDirectory(dir=self.temp.name) as temp:
            report = Path(temp)
            receipt = verify.read(REPORT / "delivery-receipt.json")
            receipt["source_tree"] = "0" * 40
            (report / "delivery-receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "Merged source tree differs"):
                verify.delivery(report, self.root, self.source_tree)

    def test_delivery_artifact_mutation(self):
        with tempfile.TemporaryDirectory(dir=self.temp.name) as temp:
            report = Path(temp)
            receipt = verify.read(REPORT / "delivery-receipt.json")
            record = receipt["receipts"]["public-readback.json"]
            readback = json.loads(record["utf8"])
            name = "vllm_ascend_pyramidkv_hust-0.4.0rc2.dev0-py3-none-any.whl"
            readback["assets"][name]["sha256"] = "0" * 64
            data = json.dumps(readback).encode()
            record.update(
                utf8=data.decode(),
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            (report / "delivery-receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "Public artifact differs"):
                verify.delivery(report, self.root, self.source_tree)

    def test_safe_names(self):
        for name in ["/tmp/escape", "../escape", "a/../escape", "a//b", "a\\b"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify.safe_name(name)

    def test_public_checksum(self):
        with tempfile.TemporaryDirectory(dir=self.temp.name) as temp:
            report = Path(temp)
            for path in REPORT.iterdir():
                if path.is_file():
                    (report / path.name).symlink_to(path)
            (report / "acceptance.json").unlink()
            (report / "acceptance.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "Public checksum mismatch"):
                verify.extract(report, report / "out")

    def test_archive_path_and_hash(self):
        for mode in ("traversal", "symlink", "digest", "duplicate"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory(dir=self.temp.name) as temp,
            ):
                report = Path(temp)
                members = {
                    "safe": {
                        "archive": "raw-common.tar.gz",
                        "bytes": 1,
                        "sha256": hashlib.sha256(b"x").hexdigest(),
                    },
                    "run": {
                        "archive": "raw-qwen35-on-558.tar.gz",
                        "bytes": 1,
                        "sha256": hashlib.sha256(b"x").hexdigest(),
                    },
                }
                for archive, name in [
                    ("raw-common.tar.gz", "safe"),
                    ("raw-qwen35-on-558.tar.gz", "run"),
                ]:
                    with tarfile.open(report / archive, "w:gz") as tar:
                        member = tarfile.TarInfo(
                            "../escape"
                            if mode == "traversal" and name == "safe"
                            else name
                        )
                        member.size = 1
                        if mode == "symlink" and name == "safe":
                            member.type = tarfile.SYMTYPE
                            member.linkname = "/tmp/escape"
                            member.size = 0
                        tar.addfile(
                            member, io.BytesIO(b"y" if mode == "digest" else b"x")
                        )
                        if mode == "duplicate" and name == "safe":
                            tar.addfile(member, io.BytesIO(b"x"))
                (report / "members.json.gz").write_bytes(
                    gzip.compress(json.dumps(members).encode())
                )
                (report / "SHA256SUMS").write_text(
                    "".join(
                        hashlib.sha256(p.read_bytes()).hexdigest()
                        + "  "
                        + p.name
                        + "\n"
                        for p in sorted(report.iterdir())
                    )
                )
                with self.assertRaisesRegex(
                    ValueError, "Unsafe|Nonregular|checksum|duplicate"
                ):
                    verify.extract(report, report / "out")


if __name__ == "__main__":
    unittest.main(verbosity=2)
