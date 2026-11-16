"""Recompute completed 180s qualifications and 900s observations from raw chunks."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent


def verify():
    index = ROOT / "windows.json"
    if not index.is_file():
        return {
            "formal_900s_points": 0,
            "qualification_points": 0,
            "requests_verified": 0,
        }
    plan = json.loads((ROOT / "formal-plan.json").read_text(encoding="utf-8"))
    count = qualifications = requests = 0
    native_long_ids = None
    for item in json.loads(index.read_text(encoding="utf-8"))["windows"]:
        if "parts" in item:
            chunks = []
            for part in item["parts"]:
                content = (ROOT / part["file"]).read_bytes()
                assert len(content) == part["bytes"]
                assert hashlib.sha256(content).hexdigest() == part["sha256"]
                chunks.append(content)
            raw = b"".join(chunks)
        else:
            raw = (ROOT / item["file"]).read_bytes()
        assert (
            len(raw) == item["bytes"]
            and hashlib.sha256(raw).hexdigest() == item["sha256"]
        )
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as package:
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
                assert (
                    len(content) == record["bytes"]
                    and hashlib.sha256(content).hexdigest() == record["sha256"]
                )
            name = next(
                name for name in members if name.endswith("/result-receipt.json")
            )
            result = json.load(package.extractfile(name))
            oracle = json.load(
                package.extractfile(
                    "evidence/" + result["arm"] + "/qualification-oracle.json"
                )
            )
            assert oracle["valid"] and len(oracle["short"]) == 8
            assert len(oracle["long"]) == 2
            assert oracle["quality_scope"] == plan["quality_scope"]
            actual_short_mismatches = []
            for row in oracle["short"]:
                cold = row["cold"]["response"]["choices"][0]["token_ids"]
                warm = row["warm"]["response"]["choices"][0]["token_ids"]
                assert row["equal"] == (cold == warm)
                if cold != warm:
                    actual_short_mismatches.append(row["index"])
            assert actual_short_mismatches == oracle["short_repeatability_mismatches"]
            assert (
                actual_short_mismatches
                == item["sampled_quality"]["short_repeatability_mismatches"]
            )
            long_ids = []
            for row in oracle["long"]:
                samples = [
                    row[mode]["response"]["choices"][0]["token_ids"]
                    for mode in ["cold", "warm", "restored"]
                ]
                assert samples[0] == samples[1] == samples[2]
                assert len(samples[0]) == 32
                long_ids.append(samples[0])
                if result["arm"].startswith("pegaflow"):
                    assert row["mechanism_delta"]["pega_load_success_total"] > 0
                    assert (
                        row["mechanism_delta"]["pega_load_failure_total"]
                        == row["mechanism_delta"]["pega_save_failure_total"]
                        == 0
                    )
            if result["arm"] == "native" and native_long_ids is None:
                native_long_ids = long_ids
            assert long_ids == native_long_ids
            prefix = name.removesuffix("result-receipt.json")
            metadata = json.load(package.extractfile(prefix + "server-metadata.json"))
            assert metadata["runtime_identity"] == plan["runtime_identity"]
            assert (
                metadata["physical_npus"] == [3, 4] and metadata["serving_chips"] == 2
            )
            assert result["valid"] and result["failed_requests"] == 0
            duration = 180 if result["qualification"] else 900
            assert result["measurement_seconds"] == duration
            rows = [
                json.loads(line)
                for line in package.extractfile(prefix + "client/requests.jsonl")
                if line.strip()
            ]
            for row in rows:
                assert row["success"] and not row["error"]
                assert (
                    len(row["token_ids"])
                    == row["usage"]["completion_tokens"]
                    == row["expected_output_tokens"]
                    == sum(n for _, n in row["chunks"])
                )
            tokens = sum(
                n for row in rows for at, n in row["chunks"] if 0 <= at < duration
            )
            assert tokens == result["observed_output_tokens_in_window"]
            assert abs(tokens / duration - result["output_tokens_per_second"]) < 1e-9
            assert (
                abs(tokens / duration / 2 - result["output_tokens_per_second_per_chip"])
                < 1e-9
            )
            assert result["arm"] == item["result"]["arm"]
            assert result["concurrency"] == item["result"]["concurrency"]
            assert (
                result["output_tokens_per_second"]
                == item["result"]["output_tokens_per_second"]
            )
            if result["qualification"]:
                qualifications += 1
            else:
                assert result["concurrency"] in plan["cells"]
                count += 1
            requests += len(rows)
    return {
        "formal_900s_points": count,
        "qualification_points": qualifications,
        "requests_verified": requests,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
