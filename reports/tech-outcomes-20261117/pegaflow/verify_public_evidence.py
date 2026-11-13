"""Verify the public capsule and optionally recompute existing archive observations."""

import argparse
import gzip
import hashlib
import json
import math
import runpy
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(archive=None, native_part=None):
    manifest = read("manifest.json")
    expected = {record["path"] for record in manifest["files"]}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != ROOT / "manifest.json"
    }
    assert actual == expected, "Public capsule file inventory changed"
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        assert path.stat().st_size == record["bytes"], record["path"]
        assert digest(path) == record["sha256"], record["path"]
    patch = gzip.decompress(
        (ROOT / "config/numa-sparse-container.patch.gz").read_bytes()
    )
    assert (
        hashlib.sha256(patch).hexdigest()
        == read("config/compatibility-patch.json")["patch_sha256"]
    )

    index = read("historical-index.json")
    assert index["not_new_performance_evidence"] is True
    assert [point["concurrency"] for point in index["points"]] == [1, 2, 4, 8, 16]
    ratios = []
    for point in index["points"]:
        baseline = point["native"]["summary"]
        candidate = point["candidate"]["summary"]
        for summary in [baseline, candidate]:
            assert summary["valid"] and summary["failed_requests"] == 0
            assert summary["measurement_seconds"] == 900
            assert math.isclose(
                summary["observed_output_tokens_in_window"] / 900,
                summary["output_tokens_per_second"],
                abs_tol=1e-9,
            )
        ratio = (
            candidate["output_tokens_per_second"] / baseline["output_tokens_per_second"]
        )
        ratios.append(ratio)
        assert math.isclose(
            (ratio - 1) * 100,
            point["published_ratio_change_percent_recomputed"],
            abs_tol=1e-9,
        )
        assert point["native"]["full_request_stream_independently_retrieved"] is False
        assert point["native"]["seed"] != point["candidate"]["seed"]
        assert (
            point["native"]["prepared_workload_sha256"]
            != point["candidate"]["prepared_workload_sha256"]
        )
    assert ratios[0] < 1, "C1 regression must remain visible"
    aggregate = (math.prod(ratios) ** 0.2 - 1) * 100
    assert math.isclose(
        aggregate,
        index["published_geometric_ratio_change_percent_recomputed"],
        abs_tol=1e-9,
    )
    assert read("config/attempted-plan.json")["formal_points_completed"] == 0
    interruption = read("diagnostics/resource-interruption-extract.json")
    assert interruption["formal_900s_points_completed"] == 0
    assert interruption["full_remote_evidence_archive_retrieved"] is False
    comparison = read("diagnostics/token-comparison-extract.json")
    assert comparison["valid"] is False
    failed = comparison["comparison"][4]
    assert failed["prompt_tokens"] == 1637 and failed["first_difference"] == 20
    assert (
        len(failed["native_ids"])
        == len(failed["cold_ids"])
        == len(failed["warm_ids"])
        == 32
    )
    assert failed["native_ids"] == failed["warm_ids"] != failed["cold_ids"]

    sources = read("sources.json")
    requests = 0
    if archive:
        custody = sources["historical_candidate_archive"]
        assert archive.stat().st_size == custody["bytes"]
        assert digest(archive) == custody["sha256"]
        with tarfile.open(archive) as package:
            for point in index["points"]:
                candidate = point["candidate"]
                prefix = (
                    custody["archive_member_prefix"] + candidate["archive_directory"]
                )
                total = count = 0
                for line in package.extractfile(prefix + "requests.jsonl"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assert row["success"] and not row["error"]
                    assert (
                        len(row["token_ids"])
                        == row["usage"]["completion_tokens"]
                        == row["expected_output_tokens"]
                        == sum(n for _, n in row["chunks"])
                    )
                    total += sum(n for at, n in row["chunks"] if 0 <= at < 900)
                    count += 1
                assert count == candidate["requests_verified"]
                assert total == candidate["window_tokens_recomputed"]
                requests += count
        assert requests == index["candidate_requests_verified"] == 1957
    if native_part:
        custody = sources["native_public_metadata"]
        assert native_part.stat().st_size == custody["part_bytes"]
        assert digest(native_part) == custody["part_sha256"]
        raw = gzip.decompress(native_part.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == custody["reconstructed_sha256"]
        public = json.loads(raw)
        for point in index["points"]:
            baseline = point["native"]
            position = int(baseline["public_metadata_json_pointer"].rsplit("/", 1)[1])
            original = public["runs"][position]
            assert original["point_id"] == baseline["point_id"]
            assert original["summary"] == baseline["summary"]
            assert original["client"]["started_at_unix"] == baseline["started_at_unix"]
            assert (
                original["requests_artifact_sha256"]
                == baseline["requests_artifact_sha256_recorded_in_public_metadata"]
            )
    restart = ROOT / "restart-02/verify.py"
    restart_result = (
        runpy.run_path(str(restart))["verify"]() if restart.is_file() else None
    )
    return {
        "public_files_verified": len(expected),
        "historical_candidate_requests_recomputed": requests,
        "native_public_metadata_verified": native_part is not None,
        "native_full_request_stream_verified": False,
        "new_formal_900s_points": 0,
        "new_speedup_claim": False,
        "restart02_diagnostics": restart_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--native-publication-part", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.native_publication_part), indent=2))


if __name__ == "__main__":
    main()
