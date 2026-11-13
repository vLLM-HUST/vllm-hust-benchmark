"""Verify byte-preserved completed diagnostics; do not promote qualifications to scores."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent


def verify():
    capsules = json.loads((ROOT / "capsules.json").read_text(encoding="utf-8"))
    results = {}
    counts = {}
    for arm, descriptor in capsules.items():
        path = ROOT / descriptor["file"]
        raw = path.read_bytes()
        assert len(raw) == descriptor["bytes"]
        assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
        with tarfile.open(path) as package:
            members = {member.name: member for member in package.getmembers()}
            assert all(
                member.isfile()
                and not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts
                for name, member in members.items()
            )
            manifest = json.load(package.extractfile("manifest.json"))
            assert set(members) == {record["path"] for record in manifest["files"]} | {
                "manifest.json"
            }
            for record in manifest["files"]:
                content = package.extractfile(record["path"]).read()
                assert len(content) == record["bytes"]
                assert hashlib.sha256(content).hexdigest() == record["sha256"]
            prefix = "diagnostics/repetition-and-restore/" + arm + "-diagnostic/"
            result = json.load(package.extractfile(prefix + "result.json"))
            assert (
                len(result["short_pairs"]) == 10
                and result["short_unique_sequences"] == 1
            )
            assert all(pair["equal"] for pair in result["short_pairs"])
            assert len(result["long_cases"]) == 2
            for name in members:
                if name.startswith(prefix + "responses/") and name.endswith(".json"):
                    record = json.load(package.extractfile(name))
                    request, response = record["request"], record["response"]
                    assert (
                        len(response["choices"][0]["token_ids"])
                        == response["usage"]["completion_tokens"]
                        == request["max_tokens"]
                        == 32
                    )
                    assert response["usage"]["prompt_tokens"] == len(request["prompt"])
            for case in result["long_cases"]:
                assert case["cold_equals_warm"] and case["warm_equals_replay"]
                assert case["reset"]["http_status"] == 200
                if arm == "pegaflow":
                    assert case["mechanism_delta"]["pega_load_success_total"] > 0
                    assert (
                        case["mechanism_delta"]["pega_load_failure_total"]
                        == case["mechanism_delta"]["pega_save_failure_total"]
                        == 0
                    )
            point = result["qualification"]
            assert (
                point["qualification"]
                and point["valid"]
                and point["failed_requests"] == 0
            )
            assert point["measurement_seconds"] == 60
            rows = [
                json.loads(line)
                for line in package.extractfile(
                    "evidence/"
                    + arm
                    + "-diagnostic/qualification-c2-60s/client/requests.jsonl"
                )
                if line.strip()
            ]
            for row in rows:
                assert row["success"] and not row["error"]
                assert (
                    len(row["token_ids"])
                    == row["usage"]["completion_tokens"]
                    == row["expected_output_tokens"]
                    == sum(count for _, count in row["chunks"])
                )
            tokens = sum(
                count for row in rows for at, count in row["chunks"] if 0 <= at < 60
            )
            assert tokens == point["observed_output_tokens_in_window"]
            assert abs(tokens / 60 - point["output_tokens_per_second"]) < 1e-9
            results[arm] = result
            counts[arm] = {
                "files_verified": len(manifest["files"]),
                "short_requests": 20,
                "long_prefixes": 2,
                "qualification_requests": len(rows),
            }
    n, p = results["native"], results["pegaflow"]
    assert all(
        pair[mode]["token_ids"] == n["short_pairs"][0]["cold"]["token_ids"]
        for pair in p["short_pairs"]
        for mode in ["cold", "warm"]
    )
    for index, case in enumerate(p["long_cases"]):
        assert case["prompt_sha256"] == n["long_cases"][index]["prompt_sha256"]
        for mode in ["cold", "local_warm", "after_local_reset"]:
            assert case[mode]["token_ids"] == n["long_cases"][index][mode]["token_ids"]
    plan = json.loads((ROOT / "formal-plan.json").read_text(encoding="utf-8"))
    assert plan["cells"] == [1, 2, 4, 8, 16] and plan["seconds_per_cell"] == 900
    assert (
        hashlib.sha256((ROOT / "run_formal.py.source").read_bytes()).hexdigest()
        == plan["driver_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / "formal_runtime.py.source").read_bytes()).hexdigest()
        == plan["runtime_helper_sha256"]
    )
    return {
        "capsules": counts,
        "cross_arm_sampled_tokens_equal": True,
        "candidate_external_restores_verified": 2,
        "classification": "diagnostic/qualification; not 900s performance evidence",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
