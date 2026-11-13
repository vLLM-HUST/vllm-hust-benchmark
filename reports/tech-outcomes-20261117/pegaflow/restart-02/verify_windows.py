"""Recompute completed 180s qualifications and 900s observations from raw chunks."""

import hashlib
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
    for item in json.loads(index.read_text(encoding="utf-8"))["windows"]:
        archive = ROOT / item["file"]
        raw = archive.read_bytes()
        assert (
            len(raw) == item["bytes"]
            and hashlib.sha256(raw).hexdigest() == item["sha256"]
        )
        with tarfile.open(archive) as package:
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
