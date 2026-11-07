"""Check positional alignment before grading a saved fixed NOMAD submission."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


COMPETITION = "nomad2018-predict-transparent-conductors"
PUBLIC_TEST_MD5 = "be4e478c19d962eae8ffde95163fe8b1"
TARGETS = ("formation_energy_ev_natom", "bandgap_energy_ev")
MAX_FILE_BYTES = 1024 * 1024


def read_limited(path):
    if not path.is_file():
        raise ValueError("regular CSV file required")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("CSV size is outside the fixed-profile limit")
    return raw


def table(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    names = reader.fieldnames or []
    if len(names) != len(set(names)) or "id" not in names:
        raise ValueError("CSV must contain unique column names and an id column")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("ragged CSV row")
    return names, rows


def audit(test_path, submission_path, *, expected_md5=PUBLIC_TEST_MD5, count=240):
    test_raw = read_limited(test_path)
    if hashlib.md5(test_raw, usedforsecurity=False).hexdigest() != expected_md5:
        raise ValueError("public test CSV does not match the publisher checksum")
    test_names, test = table(test_raw)
    if set(TARGETS).intersection(test_names):
        raise ValueError("the public test CSV must not contain grading labels")
    submission_raw = read_limited(submission_path)
    names, rows = table(submission_raw)
    if not set(TARGETS).issubset(names):
        raise ValueError("submission is missing a prediction column")
    expected_ids = [row["id"] for row in test]
    if len(expected_ids) != count or len(set(expected_ids)) != count:
        raise ValueError("unexpected public test row identity")
    if [row["id"] for row in rows] != expected_ids:
        raise ValueError("submission IDs must match the public test order exactly")
    try:
        finite = all(math.isfinite(float(row[key])) for row in rows for key in TARGETS)
    except ValueError:
        finite = False
    if not finite:
        raise ValueError("all predictions must be finite numbers")
    return {
        "competition_id": COMPETITION,
        "submission_rows": len(rows),
        "test_order_matches": True,
        "finite_predictions": True,
        "submission_sha256": hashlib.sha256(submission_raw).hexdigest(),
        "grading_executed": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-test", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.public_test, args.submission), indent=2))


if __name__ == "__main__":
    main()
