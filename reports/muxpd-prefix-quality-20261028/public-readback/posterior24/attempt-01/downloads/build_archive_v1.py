"""Build only explicit terminal-subset/CPU/audit files; never walk sibling runs."""

import gzip
import io
import json
import os
import stat
import tarfile
from pathlib import Path

from verify_archive_v1 import safe_name, sha_file, verify

BASE = Path("/root/workspace/muxpd-quality-layout-20261028")
STAGING = Path(__file__).resolve().parent
RUN = BASE / "quality-runs/prefix-quality-losses-01"
QUALITY = BASE / "quality-v1"
AUDIT = BASE / "prefix-quality-losses-independent-audit-v1"
NAME = "muxpd-prefix-quality-losses-01-20261028-v1"
EXCLUDE = {"compiler-cache", "__pycache__", ".pytest_cache", ".ruff_cache"}
CPU_FILES = (
    "cpu-tests-v1.log",
    "ruff-v1.log",
    "format-v1.log",
    "dry-run-losses-v1.json",
    "dry-run-controller-losses-v1.json",
    "root-dry-run-losses-v1.json",
    "ready-manifest-v1.json",
    "freeze-manifest-v1.json",
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def files_under(root):
    # Exact terminal root only. Prune caches before traversal. Never enumerate
    # BASE/quality-runs or a source parent containing another run.
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in EXCLUDE)
        for name in directories:
            if (Path(current) / name).is_symlink():
                raise ValueError("Symlink directory rejected")
        for name in sorted(files):
            path = Path(current) / name
            if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
                raise ValueError("Only original regular files are allowed")
            yield path


def main():
    controller, run = read(RUN / "controller.json"), read(RUN / "results/run.json")
    if (
        not controller["passed"]
        or controller["exit_code"] != 0
        or not controller["finished_at"]
        or controller.get("owned_members_after_cleanup") != []
        or controller["errors"]
        or not run["passed"]
        or run["scope"] != "loss-batches"
        or not run["finished_at"]
    ):
        raise ValueError("Refuse non-terminal/failed/non-subset run")
    if sha_file(RUN / "results/run.json") != controller["terminal_run_sha256"]:
        raise ValueError("Candidate terminal run changed")
    freeze = read(RUN / "producer/freeze-manifest-v1.json")
    if freeze["files"] != controller["files"]:
        raise ValueError("Candidate freeze differs")
    for name, expected in freeze["files"].items():
        if (
            sha_file(RUN / "producer" / name) != expected
            or sha_file(QUALITY / name) != expected
        ):
            raise ValueError("Frozen producer/input differs from execution source")
    if not read(AUDIT / "summary.json")["passed"]:
        raise ValueError("Independent audit did not pass")
    mappings = [
        (p, f"{NAME}/candidate/{p.relative_to(RUN).as_posix()}")
        for p in files_under(RUN)
    ]
    mappings += [
        (QUALITY / name, f"{NAME}/support/quality-v1/{name}") for name in CPU_FILES
    ]
    mappings += [
        (p, f"{NAME}/independent-audit/{p.relative_to(AUDIT).as_posix()}")
        for p in files_under(AUDIT)
    ]
    mappings += [
        (
            STAGING / "historical-controls-reference-v1.json",
            f"{NAME}/publication/historical-controls-reference-v1.json",
        )
    ]
    entries = []
    for source, member in sorted(mappings, key=lambda x: x[1]):
        safe_name(member)
        if not stat.S_ISREG(source.lstat().st_mode) or source.is_symlink():
            raise ValueError("Source is not an ordinary file")
        entries.append(
            {
                "member": member,
                "source": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha_file(source),
            }
        )
    manifest = {
        "schema": "muxpd-prefix-subset-publication-manifest-v1",
        "archive_name": NAME + ".tar.gz",
        "status": "local_staging_not_uploaded",
        "candidate_run": "prefix-quality-losses-01",
        "scope": "Terminal posterior batches3/6/14 candidate originals, frozen producer/inputs, CPU receipts and independent audit. Historical control full run bytes are external references only. No full200 candidate result.",
        "allowed_source_roots": [str(RUN), str(QUALITY), str(AUDIT), str(STAGING)],
        "excluded_components": sorted(EXCLUDE),
        "original_member_count": len(entries),
        "embedded_manifest_member": NAME + "/MEMBER-MANIFEST-v1.json",
        "manifest_self_rule": "This list covers every original payload member. One embedded byte-identical copy of this manifest is additional and is checked separately, avoiding a self-hash cycle.",
        "historical_native_raw_full_runs_included": False,
        "full200_candidate_outputs_included": False,
        "original_200_case_inputs_included_for_frozen_plan_identity": True,
        "members": entries,
    }
    manifest_path = STAGING / (NAME + ".manifest.json")
    archive = STAGING / (NAME + ".tar.gz")
    write(manifest_path, manifest)
    with archive.open("xb") as output:
        with gzip.GzipFile(
            filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6
        ) as zipped:
            with tarfile.open(
                fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT
            ) as tar:
                for entry in entries:
                    info = tarfile.TarInfo(entry["member"])
                    info.size, info.mode, info.mtime = entry["bytes"], 0o644, 0
                    with Path(entry["source"]).open("rb") as source:
                        tar.addfile(info, source)
                payload = manifest_path.read_bytes()
                info = tarfile.TarInfo(manifest["embedded_manifest_member"])
                info.size, info.mode, info.mtime = len(payload), 0o644, 0
                tar.addfile(info, io.BytesIO(payload))
    # One full readback checks exact member set, all original bytes, safe paths
    # and original-source identity after the copy. No extraction or execution.
    verification = verify(archive, manifest_path, compare_sources=True)
    write(STAGING / "local-verification-v1.json", verification)
    record = {
        "schema": "muxpd-prefix-subset-publication-record-v1",
        "status": "local_verified_not_uploaded",
        "archive": verification["archive"],
        "manifest": verification["manifest"],
        "verification": {
            "name": "local-verification-v1.json",
            "sha256": sha_file(STAGING / "local-verification-v1.json"),
        },
        "terminal_run_sha256": sha_file(RUN / "results/run.json"),
        "assessment_sha256": sha_file(RUN / "assessment.json"),
        "freeze_sha256": sha_file(RUN / "producer/freeze-manifest-v1.json"),
        "independent_audit_summary_sha256": sha_file(AUDIT / "summary.json"),
        "historical_controls_reference_sha256": sha_file(
            STAGING / "historical-controls-reference-v1.json"
        ),
        "builder_sha256": sha_file(Path(__file__)),
        "verifier_sha256": sha_file(STAGING / "verify_archive_v1.py"),
        "original_member_count": len(entries),
        "historical_native_raw_full_runs_included": False,
        "full200_candidate_outputs_included": False,
        "remote_verified": False,
    }
    write(STAGING / "archive-record-v1.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
