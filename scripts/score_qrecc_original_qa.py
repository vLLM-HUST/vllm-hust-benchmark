"""Score saved QReCC answers using unchanged publisher QA functions.

This does not retrieve passages, generate answers, or prove end-to-end delivery.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

AUTHOR_REVISION = "1d174464290790c0198b474ad022952ecd83b76b"  # pragma: allowlist secret - public author revision
SCORER_SHA = "f8fe4c1c163004953791983fe141363566b741851f42d7f3fa473646ed672165"  # pragma: allowlist secret - public author source checksum
TEST_SHA = "a4d8ef686827feb00c90997e0d6ff135928bf6f4bcb26fb523b5fac025d58f84"  # pragma: allowlist secret - public original test checksum


def load_author(path):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCORER_SHA:
        raise ValueError("unchanged pinned publisher evaluate_qa.py required")
    # Execute exactly the verified bytes, not a second unverified loader read.
    spec = importlib.util.spec_from_loader("qrecc_pinned_qa", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def score(*, data_file, scorer_file, predictions_file, output):
    paths = [Path(data_file), Path(scorer_file), Path(predictions_file)]
    raw = [path.read_bytes() for path in paths]
    hashes = [hashlib.sha256(value).hexdigest() for value in raw]
    if hashes[0] != TEST_SHA:
        raise ValueError("unchanged original QReCC test required")
    author = load_author(paths[1])
    rows = json.loads(raw[0])
    predictions = json.loads(raw[2])
    if not isinstance(predictions, list) or len(predictions) < 3:
        raise ValueError("at least three preselected original predictions required")
    indices = [item.get("original_row_index") for item in predictions]
    if any(
        type(index) is not int or not 0 <= index < len(rows) for index in indices
    ) or indices != sorted(set(indices)):
        raise ValueError("distinct original row indices in original order required")
    results = []
    for item, index in zip(predictions, indices):
        row = rows[index]
        if (
            item.get("Conversation_no") != row["Conversation_no"]
            or item.get("Turn_no") != row["Turn_no"]
            or not isinstance(item.get("prediction"), str)
            or not isinstance(row.get("Answer"), str)
            or set(item)
            != {"original_row_index", "Conversation_no", "Turn_no", "prediction"}
        ):
            raise ValueError(
                "original task identity and text-only prediction contract required"
            )
        results.append(
            {
                "original_row_index": index,
                "Conversation_no": row["Conversation_no"],
                "Turn_no": row["Turn_no"],
                "exact_match": author.compute_exact(row["Answer"], item["prediction"]),
                "f1": author.compute_f1(row["Answer"], item["prediction"]),
            }
        )
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != expected
        for path, expected in zip(paths, hashes)
    ):
        raise ValueError("scoring input identity changed during evaluation")
    output = Path(output)
    if any(output.resolve().is_relative_to(path.resolve()) for path in paths):
        raise ValueError("fresh output separate from scoring inputs required")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "QRECC",
        "status": "SAVED_ANSWERS_SCORED_WITH_PUBLISHER_QA_FUNCTIONS",
        "author_revision": AUTHOR_REVISION,
        "test_sha256": hashes[0],
        "scorer_sha256": hashes[1],
        "predictions_sha256": hashes[2],
        "selected_original_rows": indices,
        "samples": len(results),
        "exact_match": sum(item["exact_match"] for item in results) / len(results),
        "f1": sum(item["f1"] for item in results) / len(results),
        "metric_scale": "0_to_1",
        "tasks": results,
        "model_calls": 0,
        "retrieval_executed": False,
        "generation_provenance_verified": False,
        "source_checkbox_eligible": False,
        "proof_boundary": "Saved-answer scoring only; does not verify how predictions were generated, original retrieval coverage, or end-to-end benchmark completion.",
    }
    target = output / "receipt-private.json"
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    target.chmod(0o600)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--scorer-file", type=Path, required=True)
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(**vars(args))
    print(
        json.dumps(
            {key: result[key] for key in ("status", "samples", "exact_match", "f1")}
        )
    )


if __name__ == "__main__":
    main()
