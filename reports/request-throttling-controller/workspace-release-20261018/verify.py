"""Verify only owned lifecycle evidence; no model performance claims."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root, archive=None):
    root = Path(root)
    if (root / "SHA256SUMS").exists():
        for line in (root / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert Path(name).name == name
            assert sha((root / name).read_bytes()) == expected
    proof = json.loads((root / "receipt.json").read_text())
    assert proof["status"] == "THREE_WORKSPACES_STOPPED_AND_RELEASE_VERIFIED"
    assert len(proof["rows"]) == 3 and proof["released_npu_requests"] == 6
    for row in proof["rows"]:
        assert row["workspace_status"] == "stopped" and row["raycluster_absent"]
        assert row["pod_count_after"] == row["owned_worker_process_count_after"] == 0
        assert row["released_npu_requests"] == 2
        assert (
            row["persistent_data_retained"] and row["target_cards_no_running_processes"]
        )
    if archive:
        assert sha(Path(archive).read_bytes()) == proof["private_raw_sha256"]
        with tarfile.open(archive) as tar:
            files = {}
            for entry in tar.getmembers():
                path = Path(entry.name)
                assert not path.is_absolute() and ".." not in path.parts
                assert entry.isfile()
                files[entry.name] = tar.extractfile(entry).read()
        manifest_bytes = files["artifact-manifest.json"]
        assert sha(manifest_bytes) == proof["private_manifest_sha256"]
        manifest = json.loads(manifest_bytes)
        assert set(files) == set(manifest) | {"artifact-manifest.json"}
        for name, expected in manifest.items():
            assert sha(files[name]) == expected

        def read(name):
            return json.loads(files[name])

        before_workspaces = read("coder-pre-cleanup-private.json")
        after_workspaces = read("coder-post-cleanup-private.json")
        intent = read("cleanup-stop-intent.json")["authorized_scope"]
        assert len(intent) == 3
        after_kubernetes = read("cleanup-preflight-private/kubernetes-after.json")
        for row in proof["rows"]:
            label = row["dataset"]
            prefix = "cleanup-preflight-private/"
            before = read(prefix + label + ".json")
            binding = read(prefix + label + "-host-binding.json")
            after_host = read(prefix + label + "-host-after.json")
            service = read(prefix + label + "-service-log-snapshot.json")
            assert (
                binding["boot_id_sha256"]
                == after_host["boot_id_sha256"]
                == service["boot_id_sha256"]
            )
            assert binding["persistent_root"] == after_host["persistent_root"]
            assert (
                binding["persistent_root_exists"]
                and after_host["persistent_root_exists"]
            )
            assert len(binding["processes"]) == 2
            assert not any(after_host["owned_pids_still_exist"].values())
            assert set(binding["processes"]) == set(
                after_host["owned_pids_still_exist"]
            )
            pods = before["pods"]["items"]
            assert len(pods) == 2
            assert sorted(sha(p["metadata"]["uid"].encode()) for p in pods) == sorted(
                row["pod_identity_sha256"]
            )
            pod_ids = [p["metadata"]["uid"] for p in pods]
            container_ids = [
                c["containerID"].split("://")[-1]
                for p in pods
                for c in p["status"]["containerStatuses"]
            ]
            for process in binding["processes"].values():
                group = process["cgroup"]
                assert any(
                    identity in group or identity.replace("-", "_") in group
                    for identity in pod_ids + container_ids
                )
            requests = sum(
                int(value)
                for p in pods
                for c in p["spec"]["containers"]
                for key, value in c.get("resources", {}).get("requests", {}).items()
                if "Ascend" in key
            )
            assert requests == row["released_npu_requests"]
            workspace_id = next(
                uid
                for uid in intent.values()
                if sha(uid.encode()) == row["workspace_identity_sha256"]
            )
            old = next(w for w in before_workspaces if w["id"] == workspace_id)
            new = next(w for w in after_workspaces if w["id"] == workspace_id)
            assert (
                old["latest_build"]["status"] == "running"
                and new["latest_build"]["status"] == "stopped"
            )
            assert old["owner_name"] == new["owner_name"] and old["name"] == new["name"]
            stop = read(prefix + old["name"] + "-stop-result.json")
            assert stop["returncode"] == 0 and stop["id"] == workspace_id
            cluster_name = pods[0]["metadata"]["labels"]["ray.io/cluster"]
            observations = [
                x
                for x in after_kubernetes.values()
                if not x["cluster"] and not x["pods"]["items"]
            ]
            assert len(observations) == 3 and cluster_name
            for job in before["runtime"]["jobs"]:
                if job["submission_id"].endswith("--queue"):
                    assert job["status"] in ["SUCCEEDED", "STOPPED"]
            pid_set = set(binding["processes"])
            matches = []
            for name, data in files.items():
                if name.endswith("-npu-before.log"):
                    tuples = re.findall(
                        r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", data.decode()
                    )
                    cards = [card for card, _, pid in tuples if pid in pid_set]
                    if len(cards) == 2:
                        matches.append((name, cards))
            assert len(matches) == 1
            name, cards = matches[0]
            after_npu = files[name.replace("-before.log", "-after.log")].decode()
            assert all(
                "No running processes found in NPU " + card in after_npu
                for card in cards
            )
    return {
        "owned_release": "PASS",
        "private_archive_verified": bool(archive),
        "stopped_workspaces": 3,
        "released_npu_requests": 6,
        "persistent_data_retained": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive")
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).parent, args.archive)))
