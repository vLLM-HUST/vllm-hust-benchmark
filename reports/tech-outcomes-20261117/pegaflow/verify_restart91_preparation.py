"""Verify fresh91 source, actual resource/CPU and full frozen input admission."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def verify():
    root = Path(__file__).parent
    for name, manifest, checksum, count in [
        (
            "driver-mod-v4-91",
            "source-manifest.json",
            "8e4ef4748eafd8ad0cdc6fd0c1d82df471576357cca2333ce1abc4abe168c34e",
            21,
        ),
        (
            "analysis-v4-91",
            "analysis-manifest.json",
            "782cecbbaac0757f388ece22f335dcceefb19bfb83345f79a1fa27ffbaa8b408",
            13,
        ),
    ]:
        folder = root / name
        raw = (folder / manifest).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == checksum
        members = json.loads(raw)["files"]
        assert len(members) == count
        for source, expected in members.items():
            stored = (
                "original-README.md.source"
                if source == "README.md"
                else source + ".source"
                if source.endswith(".py")
                else source
            )
            assert (
                hashlib.sha256((folder / stored).read_bytes()).hexdigest() == expected
            )
    folder = root / "cpu-resource-admission-v4-91"
    for item in read(folder / "manifest.json")["files"]:
        name = item["path"]
        stored = name + ".source" if name.endswith(".py") else name
        raw = (folder / stored).read_bytes()
        assert len(raw) == item["bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    cpu = read(folder / "actual-cpu-admission.json")
    assert cpu["phase"] == "passed"
    assert cpu["model_startups"] == cpu["model_generation_requests"] == 0
    assert read(folder / "actual-cpu-admission-wait.json")["actual_returncode"] == 0
    admission = read(folder / "pregen-input-admission.json")
    assert admission["phase"] == "passed"
    assert admission["model_startups"] == admission["generation_requests"] == 0
    assert admission["scopes"]["frontier_membership"]["tasks"] == 160
    assert admission["scopes"]["mmlu"]["tasks"] == 200
    assert admission["scopes"]["swe"]["requests"] == 64
    plan = read(root / "driver-mod-v4-91/execution-plan.json")
    assert plan["physical_npus"] == [2, 5]
    assert plan["logical_npus"] == [0, 1]
    assert plan["kv_bytes_per_rank"] == 17179869184
    return {
        "main_members": 21,
        "extra_members": 13,
        "actual_cpu_passed": True,
        "model_result": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
