"""Publisher seed-zero option permutation, not generated task content."""

import csv
import hashlib
import io
import json
from pathlib import Path
from random import Random
from zipfile import ZipFile


def seed_zero_choices(row, physical_index):
    if type(physical_index) is not int or physical_index < 0:
        raise ValueError("GPQA requires a nonnegative physical row offset")
    keys = [
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "Correct Answer",
    ]
    if any(
        not isinstance(row.get(k), str) or not row[k].strip()
        for k in ["Question", *keys]
    ):
        raise ValueError("GPQA requires original nonempty question and four answers")
    # load_examples seeds once and shuffles four choices per preceding row.
    # Reproduce that state for random-access resolve without changing global RNG.
    random = Random(0)
    dummy = [0, 1, 2, 3]
    for _ in range(physical_index):
        random.shuffle(dummy)
    choices = [row[k] for k in keys]
    random.shuffle(choices)
    return choices, choices.index(row["Correct Answer"])


def prepare_original_archive(*, data_archive, output):
    """Extract only registered original question members after byte checks."""
    from .registry import source_spec

    source = source_spec("GPQA")
    (archive_spec,) = [f for f in source["files"] if f["role"] == "source_archive"]
    archive = Path(data_archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != archive_spec["sha256"]:
        raise ValueError("publisher GPQA archive identity mismatch")
    output = Path(output)
    if output.resolve() == archive.parent.resolve() or output.exists():
        raise ValueError("fresh output directory separate from archive required")
    # Validate every member before creating any output; no archive extractall.
    verified = []
    with ZipFile(archive) as zipped:
        for asset in source["files"]:
            if asset["role"] != "data":
                continue
            password = b"deserted-untie-orchid"  # pragma: allowlist secret - password published in author README
            raw = zipped.read(asset["archive_member"], pwd=password)
            if (
                len(raw) != asset["bytes"]
                or hashlib.sha256(raw).hexdigest() != asset["sha256"]
            ):
                raise ValueError("publisher GPQA original member identity mismatch")
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            if len(rows) != asset["rows"]:
                raise ValueError("publisher GPQA row count mismatch")
            for i, row in enumerate(rows):
                seed_zero_choices(row, i)
            verified.append((asset, raw))
    output.mkdir(mode=0o700)
    for asset, raw in verified:
        # Catalog names are flat, but enforce containment independently.
        target = (output / asset["path"]).resolve()
        if target.parent != output.resolve():
            raise ValueError("GPQA output member escapes fresh directory")
        with target.open("xb") as stream:
            stream.write(raw)
    receipt = {
        "source_id": "GPQA",
        "revision": source["revision"],
        "status": "ORIGINAL_CSV_MEMBERS_PREPARED_NOT_NATIVE_EXECUTION",
        "files": [
            {k: a[k] for k in ["path", "bytes", "sha256", "rows"]} for a, _ in verified
        ],
        "expert_metadata_is_not_a_question_benchmark": True,
        "raw_payloads_published": False,
        "model_calls": 0,
    }
    (output / "preparation-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt
