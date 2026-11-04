"""Reject incomplete or mismatched converted answers before author FAC scoring."""

import argparse
import json
from pathlib import Path

from prepare_stabletoolbench_fixed_g1 import IDS


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def check(rows, ids, answers):
    keys = list(map(str, IDS))
    if [row["query_id"] for row in rows] != IDS or list(ids) != keys:
        raise ValueError("fixed task IDs or FAC ID order changed")
    if set(answers) != set(keys):
        raise ValueError("converted answers must contain exactly the three fixed IDs")
    empty = []
    for key, row in zip(keys, rows):
        answer = answers[key]
        if answer["query"] != row["query"]:
            raise ValueError(f"converted query does not match original row {key}")
        final = answer["answer"]["final_answer"]
        if not isinstance(final, str):
            raise ValueError(f"final_answer must be a string for {key}")
        if not final:
            empty.append(key)
    return {
        "checked_ids": IDS,
        "empty_final_answer_ids": empty,
        "scoring_executed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("instructions", "ids", "answers"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = check(
            *(read_json(p) for p in (args.instructions, args.ids, args.answers))
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
