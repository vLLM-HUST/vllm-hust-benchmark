"""Verify the completed cohort's exact workload, environment and shutdown bundle."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent


def verify():
    descriptor_path = ROOT / "final-sources.json"
    if not descriptor_path.is_file():
        return {"status": "not-collected; campaign may still be running"}
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    chunks = []
    for part in descriptor["parts"]:
        raw = (ROOT / part["file"]).read_bytes()
        assert len(raw) == part["bytes"]
        assert hashlib.sha256(raw).hexdigest() == part["sha256"]
        chunks.append(raw)
    raw = b"".join(chunks)
    assert len(raw) == descriptor["bytes"]
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    plan = json.loads((ROOT / "formal-plan.json").read_text(encoding="utf-8"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as package:
        members = {item.name: item for item in package.getmembers()}
        assert all(
            item.isfile()
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            for name, item in members.items()
        )
        manifest = json.load(package.extractfile("manifest.json"))
        assert set(members) == {item["path"] for item in manifest["files"]} | {
            "manifest.json"
        }
        for item in manifest["files"]:
            data = package.extractfile(item["path"]).read()
            assert len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
        controlled = descriptor.get("schema") == "pegaflow-final-source-custody/v2"
        if controlled:
            assert "root/prepared-qwen35.json" not in members
            assert "root/diagnostic-long-prompts.json" not in members
            custody = json.load(
                package.extractfile("root/prepared-workload-custody.json")
            )
            assert (
                custody["prepared_workload_sha256"]
                == descriptor["prepared_workload_sha256"]
                == plan["prepared_workload_sha256"]
            )
            assert custody["sessions"] == 8 and custody["turns"] == 360
            assert custody["long_diagnostic_tokens"] == [7551, 8712]
            assert custody["public_input_reconstruction"] is False
            assert (
                custody["controlled_original_archive"]
                == descriptor["controlled_originals"]
            )
            assert {item["path"] for item in custody["restricted_members"]} == {
                "root/prepared-qwen35.json",
                "root/diagnostic-long-prompts.json",
            }
            # The public verifier checks the binding and custody index; it does
            # not claim to read back the restricted original input bytes.
            restricted = {item["path"]: item for item in custody["restricted_members"]}
            assert (
                restricted["root/prepared-qwen35.json"]["sha256"]
                == plan["prepared_workload_sha256"]
            )
        else:
            prepared = package.extractfile("root/prepared-qwen35.json").read()
            assert (
                hashlib.sha256(prepared).hexdigest()
                == descriptor["prepared_workload_sha256"]
                == plan["prepared_workload_sha256"]
            )
            workload = json.loads(prepared)
            assert len(workload["sessions"]) == 8
            assert sum(len(session["turns"]) for session in workload["sessions"]) == 360
            long_prompts = json.load(
                package.extractfile("root/diagnostic-long-prompts.json")
            )
            assert [len(item["input_ids"]) for item in long_prompts] == plan[
                "qualification"
            ]["long_prompt_tokens"]
        assert json.load(package.extractfile("root/formal-plan.json")) == plan
        status = json.load(package.extractfile("root/campaign-status.json"))
        assert status["phase"] == "complete"
        assert (
            package.extractfile("final/npu-release.txt")
            .read()
            .count(b"No running processes found in NPU")
            == 2
        )
        packages = json.load(package.extractfile("final/python-packages.json"))
        assert packages and all(item["name"] and item["version"] for item in packages)
    return {
        "status": "verified",
        "files_verified": len(manifest["files"]),
        "workload_sha256": descriptor["prepared_workload_sha256"],
        "sessions": 8,
        "turns": 360,
        "owned_npus_released": True,
        "original_workload_bytes_publicly_reverified": not controlled,
        "original_workload_custody_index_verified": controlled,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
