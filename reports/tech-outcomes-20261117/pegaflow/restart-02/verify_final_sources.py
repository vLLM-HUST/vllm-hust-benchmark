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
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
