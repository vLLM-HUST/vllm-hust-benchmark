"""Acquire or verify pinned ScienceWorld author runtime; never run a model."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "e8216d6044e8e39be9fcb185e3b2dfb602584b52"  # pragma: allowlist secret - public publisher Git identity
FILES = [
    {
        "path": "LICENSE",
        "sha": "7d48151e35f0b35d8b5c272686ee60bafe9605ea",  # pragma: allowlist secret - public publisher Git identity
        "size": 11394,
    },
    {
        "path": "MANIFEST.in",
        "sha": "12075063287e25692a39ad9dace1e39f60460f1c",  # pragma: allowlist secret - public publisher Git identity
        "size": 170,
    },
    {
        "path": "README.md",
        "sha": "703769290b34a1dcca1310ce3d882f54f0601576",  # pragma: allowlist secret - public publisher Git identity
        "size": 8936,
    },
    {
        "path": "examples/human.py",
        "sha": "d0cb8d7f08a13aa9c96b10d553c00dff22bc654c",  # pragma: allowlist secret - public publisher Git identity
        "size": 8524,
    },
    {
        "path": "examples/random_agent.py",
        "sha": "07d7db4e9a93767cd95b22195f205b651386019a",  # pragma: allowlist secret - public publisher Git identity
        "size": 9147,
    },
    {
        "path": "requirements.txt",
        "sha": "4464d310fcc7d66be40469d67d0e6ec943dbb92d",  # pragma: allowlist secret - public publisher Git identity
        "size": 5,
    },
    {
        "path": "scienceworld/__init__.py",
        "sha": "f59eb1b0f8ca9044c49bd4d574da8a7cabb09fcd",  # pragma: allowlist secret - public publisher Git identity
        "size": 192,
    },
    {
        "path": "scienceworld/constants.py",
        "sha": "5e46657e2f2a4f512060e31aba92ac2a9a36b9eb",  # pragma: allowlist secret - public publisher Git identity
        "size": 1123,
    },
    {
        "path": "scienceworld/object_type_ids.tsv",
        "sha": "c57e5f3f61a541772831c036f98fa9d87d946912",  # pragma: allowlist secret - public publisher Git identity
        "size": 10633,
    },
    {
        "path": "scienceworld/scienceworld.jar",
        "sha": "3b9d98036bfd7949c75d2454f1268ec11a9320dd",  # pragma: allowlist secret - public publisher Git identity
        "size": 7793417,
    },
    {
        "path": "scienceworld/scienceworld.py",
        "sha": "5217e5affbeccdfa09c58d40787d030bf8440cb3",  # pragma: allowlist secret - public publisher Git identity
        "size": 28103,
    },
    {
        "path": "scienceworld/tasks.json",
        "sha": "527fb6511da9868bf847406be00a84fc50c1f325",  # pragma: allowlist secret - public publisher Git identity
        "size": 4871,
    },
    {
        "path": "scienceworld/utils.py",
        "sha": "1be96cd5b7bb10cef41703fa348a332ce12120f7",  # pragma: allowlist secret - public publisher Git identity
        "size": 929,
    },
    {
        "path": "setup.py",
        "sha": "1e7ab0527918ff71f92e8a269043c376f7d1fde0",  # pragma: allowlist secret - public publisher Git identity
        "size": 1811,
    },
    {
        "path": "requirements-dev.txt",
        "sha": "b26c3a92b1d301ba9c246521ec388768b7c33884",  # pragma: allowlist secret - public publisher Git identity
        "size": 46,
    },
]


REACT_REVISION = "b8ea624231a6a8d83d78c1c12bea1b44b8ea758b"  # pragma: allowlist secret - public publisher Git identity
REACT_FILES = [
    {
        "path": "README.md",
        "sha": "9aab90d7f0921689ea3bd0320e219d556d77061c",  # pragma: allowlist secret - public publisher Git identity
        "size": 6186,
    },
    {
        "path": "baselines/ReAct_baseline/README.md",
        "sha": "e90bcb0894a329383295f160fa516496eede4b8a",  # pragma: allowlist secret - public publisher Git identity
        "size": 376,
    },
    {
        "path": "baselines/ReAct_baseline/prompt.jsonl",
        "sha": "90fd70fd0a1960ccf57dce5cc577283bcc4d0a51",  # pragma: allowlist secret - public publisher Git identity
        "size": 147757,
    },
    {
        "path": "baselines/eval_agent_react.py",
        "sha": "709a3aa651661c3c5600c88663cd1d590a4809ba",  # pragma: allowlist secret - public publisher Git identity
        "size": 9536,
    },
    {
        "path": "baselines/run_eval_react.sh",
        "sha": "ed07c290d55c22d563aad0dc2284e5af21fd09ad",  # pragma: allowlist secret - public publisher Git identity
        "size": 387,
    },
    {
        "path": "data_utils/__init__.py",
        "sha": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",  # pragma: allowlist secret - public publisher Git identity
        "size": 0,
    },
    {
        "path": "data_utils/data_utils.py",
        "sha": "09db8efb695e2a0fcef18a74e6464128cbdecfa6",  # pragma: allowlist secret - public publisher Git identity
        "size": 23792,
    },
    {
        "path": "eval_utils.py",
        "sha": "5bdbb1d846c758190a7e93b3e137bfa467779b55",  # pragma: allowlist secret - public publisher Git identity
        "size": 45310,
    },
    {
        "path": "requirements.txt",
        "sha": "5b1edaf4c5e608a2f06fbb1f80152bc4e16a5c04",  # pragma: allowlist secret - public publisher Git identity
        "size": 2916,
    },
    {
        "path": "slow_agent/local_llm.py",
        "sha": "6e1a40737588376661d058895f55b12f4ab74bc9",  # pragma: allowlist secret - public publisher Git identity
        "size": 14688,
    },
    {
        "path": "slow_agent/utils.py",
        "sha": "c6137d3bb217e27fc33700e5227603cd054e854a",  # pragma: allowlist secret - public publisher Git identity
        "size": 534,
    },
    {
        "path": "goldpaths-all.zip",
        "original_path": "data_utils/goldpaths-all.zip",
        "sha": "b5574a6d551863748a39571f14ff0e807cc75e64",  # pragma: allowlist secret - public publisher Git identity
        "size": 8436384,
    },
]


def verify(path, item):
    if path.is_symlink():
        raise ValueError("symlink is not an original asset")
    data = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(data)}".encode() + bytes([0]) + data).hexdigest()
    if len(data) != item["size"] or digest != item["sha"]:
        raise ValueError("original Git blob identity mismatch")
    return hashlib.sha256(data).hexdigest()


def prepare(
    output,
    *,
    execute=False,
    verify_only=False,
    profile="environment",
    reuse_root=None,
    gold_archive=None,
):
    if profile not in {"environment", "author-react"}:
        raise ValueError("unknown original asset profile")
    files = REACT_FILES if profile == "author-react" else FILES
    repo = (
        "SwiftSage/SwiftSage" if profile == "author-react" else "allenai/ScienceWorld"
    )
    if (reuse_root is None) != (gold_archive is None) or (
        reuse_root is not None and (profile != "author-react" or not execute)
    ):
        raise ValueError(
            "explicit author-react execute reuse needs both source root and gold archive"
        )
    result = {
        "source_id": "SCIENCEWORLD",
        "revision": REACT_REVISION if profile == "author-react" else REVISION,
        "profile": profile,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "status": "PLAN_ONLY",
        "files": files,
        "scope": "author environment assets, not LLM agent execution or scoring",
    }
    if not execute and not verify_only:
        return result
    output = Path(output).resolve()
    repository = Path(__file__).resolve().parents[1]
    if (repository / ".git").exists() and output.is_relative_to(repository):
        raise ValueError("original assets must remain outside repository")
    if execute and output.exists():
        raise ValueError("fresh output required; preserve partial work")
    if execute:
        output.mkdir(parents=True)
    checked = []
    for item in files:
        path = output / item["path"]
        if execute and reuse_root is not None:
            original = (
                Path(gold_archive)
                if item["path"] == "goldpaths-all.zip"
                else Path(reuse_root) / item["path"]
            )
            verify(original, item)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(original.read_bytes())
        elif execute:
            request = Request(
                "https://api.github.com/repos/" + repo + "/git/blobs/" + item["sha"],
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "workload-native-preparation",
                },
            )
            with urlopen(request, timeout=90) as response:
                blob = json.loads(response.read(item["size"] * 2 + 8192))
            if blob["sha"] != item["sha"] or blob["encoding"] != "base64":
                raise ValueError("publisher blob metadata mismatch")
            data = base64.b64decode(blob["content"], validate=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        checked.append({**item, "sha256": verify(path, item)})
    result.update(status="AUTHOR_RUNTIME_VERIFIED_NOT_NATIVE_DELIVERY", files=checked)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["environment", "author-react"], default="environment"
    )
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--gold-archive", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.output,
                execute=args.execute,
                verify_only=args.verify_only,
                profile=args.profile,
                reuse_root=args.reuse_root,
                gold_archive=args.gold_archive,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
