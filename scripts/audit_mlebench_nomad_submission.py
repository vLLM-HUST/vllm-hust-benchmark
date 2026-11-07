"""Check positional alignment before grading a saved fixed NOMAD submission."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re


COMPETITION = "nomad2018-predict-transparent-conductors"
PUBLIC_TEST_MD5 = (
    "be4e478c19d962eae8ffde95163fe8b1"  # pragma: allowlist secret - publisher checksum
)
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


def docker_timestamp(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("UTC Docker timestamp required")
    stamp = re.sub(r"(\.\d{6})\d+Z$", r"\1Z", value)
    return datetime.fromisoformat(stamp[:-1] + "+00:00")


def audit_terminal(start, terminal, submission_audit, grader_log):
    """Require the actual terminal run, frozen submission and the original grader."""
    if (
        start.get("original_steps") != 500
        or start.get("original_time_seconds") != 86400
        or terminal.get("status")
        != "NATIVE_AGENT_TERMINAL_AND_SAVED_SUBMISSION_GRADE_AUDIT"
    ):
        raise ValueError("original NOMAD terminal protocol required")
    exit_code = terminal.get("agent_exit_code")
    launcher_exit = terminal.get("original_launcher_exit_code")
    if (
        type(exit_code) is not int
        or exit_code not in (0, 124)
        or launcher_exit != exit_code
    ):
        raise ValueError("interrupted run or missing matching original launcher exit")
    duration = (
        docker_timestamp(terminal.get("agent_finished_at"))
        - docker_timestamp(terminal.get("agent_started_at"))
    ).total_seconds()
    if duration <= 0 or (exit_code == 124 and duration < 86400):
        raise ValueError("terminal timing does not establish the original budget")
    successes = terminal.get("response_status_counts", {}).get("200")
    requests = terminal.get("real_model_requests")
    if (
        terminal.get("agent_oom") is not False
        or type(successes) is not int
        or type(requests) is not int
        or not 0 < successes <= requests
        or terminal.get("journal_present") is not True
        or terminal.get("best_solution_present") is not True
    ):
        raise ValueError("successful real requests and Agent artifacts required")
    digest = submission_audit["submission_sha256"]
    if (
        terminal.get("submission_sha256") != digest
        or terminal.get("pregrade_exit_code") != 0
        or terminal.get("grade_exit_code") != 0
        or terminal.get("valid_submission") is not True
    ):
        raise ValueError("frozen submission identity and valid official grade required")
    results = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\s*\{", grader_log):
        try:
            value, _ = decoder.raw_decode(grader_log, match.end() - 1)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("competition_id") == COMPETITION:
            results.append(value)
    if len(results) != 1:
        raise ValueError("one original NOMAD grader report required")
    grade = results[0]
    score = grade.get("score")
    if (
        grade.get("valid_submission") is not True
        or grade.get("submission_exists") is not True
        or grade.get("is_lower_better") is not True
        or grade.get("submission_path") != "/submission.csv"
        or type(score) not in (float, int)
        or not math.isfinite(score)
        or score < 0
    ):
        raise ValueError("invalid original NOMAD score report")
    return {
        **submission_audit,
        "status": "NOMAD_FIXED_TERMINAL_AND_ORIGINAL_GRADE_AUDITED",
        "original_launcher_exit_code": launcher_exit,
        "agent_wall_seconds": duration,
        "successful_model_responses": successes,
        "official_score": score,
        "grading_executed": True,
        "manual_delivery_audit_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-test", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--start-receipt", type=Path)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--grader-log", type=Path)
    args = parser.parse_args()
    terminal_inputs = (args.start_receipt, args.terminal_receipt, args.grader_log)
    if any(terminal_inputs) and not all(terminal_inputs):
        parser.error(
            "start receipt, terminal receipt and grader log are required together"
        )
    result = audit(args.public_test, args.submission)
    if all(terminal_inputs):
        result = audit_terminal(
            json.loads(read_limited(args.start_receipt)),
            json.loads(read_limited(args.terminal_receipt)),
            result,
            read_limited(args.grader_log).decode("utf-8"),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
