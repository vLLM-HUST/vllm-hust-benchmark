"""Extract only pinned original QASPER v0.3 tasks and publisher evaluator.

No network, model calls, generated prompts or evaluation runs occur here.
Install the matching workload-data package before invoking this entrypoint.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

from vllm_hust_workload_data.qasper import question_rows
from vllm_hust_workload_data.registry import source_spec


ARCHIVES = {
    "qasper-train-dev-v0.3.tgz": (
        10835856,
        "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",  # pragma: allowlist secret - recorded official archive SHA256
    ),
    "qasper-test-and-evaluator-v0.3.tgz": (
        3865061,
        "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72",  # pragma: allowlist secret - recorded official archive SHA256
    ),
}


def prepare(train_dev_archive, test_archive, output):
    spec = source_spec("QASPER")
    paths = {
        "qasper-train-dev-v0.3.tgz": Path(train_dev_archive),
        "qasper-test-and-evaluator-v0.3.tgz": Path(test_archive),
    }
    contents = {}
    for name, path in paths.items():
        size, expected = ARCHIVES[name]
        raw = path.read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("original QASPER archive identity mismatch")
        with tarfile.open(path, "r:gz") as archive:
            for asset in spec["files"]:
                if asset["url"].rsplit("/", 1)[-1] != name:
                    continue
                matches = [
                    member
                    for member in archive.getmembers()
                    if member.name == asset["archive_member"]
                ]
                if len(matches) != 1 or not matches[0].isfile():
                    raise ValueError("missing or ambiguous original QASPER member")
                body = archive.extractfile(matches[0]).read()
                if (
                    len(body) != asset["bytes"]
                    or hashlib.sha256(body).hexdigest() != asset["sha256"]
                ):
                    raise ValueError("original QASPER member identity mismatch")
                contents[asset["path"]] = body
    if set(contents) != {asset["path"] for asset in spec["files"]}:
        raise ValueError("incomplete original QASPER asset set")
    output = Path(output).resolve()
    for relative in contents:
        if not (output / relative).resolve().is_relative_to(output):
            raise ValueError("unsafe QASPER output path")
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for relative, body in contents.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    splits = []
    for asset in spec["files"]:
        if asset["role"] != "data":
            continue
        count = sum(1 for _ in question_rows(output / asset["path"]))
        if count != asset["rows"]:
            raise ValueError("original QASPER question population mismatch")
        splits.append(
            {
                "file": asset["path"],
                "split": asset["split"],
                "original_questions": count,
            }
        )
    receipt = {
        "source_id": "QASPER",
        "status": "ORIGINAL_V03_ASSETS_EXTRACTED_NOT_NATIVE_EXECUTION",
        "revision": spec["revision"],
        "splits": splits,
        "publisher_evaluator_preserved_not_executed": True,
        "original_data_rewritten": False,
        "model_calls": 0,
        "native_benchmark_completed": False,
        "raw_payloads_published": False,
    }
    (output / "preparation-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dev-archive", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_dev_archive, args.test_archive, args.output)))


if __name__ == "__main__":
    main()
