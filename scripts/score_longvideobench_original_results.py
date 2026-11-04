"""Score saved fixed original LongVideoBench responses with installed lmms-eval."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    from lmms_eval.tasks.longvideobench.utils import (
        longvideobench_aggregate_results,
        longvideobench_process_results,
    )

    rows = json.loads(args.validation.read_text(encoding="utf-8"))
    if len(rows) != 1337:
        raise ValueError("Expected the pinned original validation split")
    random.seed(args.seed)
    results = []
    for index in range(3):
        original = rows[index]
        saved = json.loads(
            (args.responses / f"{index}.json").read_text(encoding="utf-8")
        )
        if saved["id"] != original["id"] or saved["reference"] != chr(
            65 + original["correct_choice"]
        ):
            raise ValueError(f"Saved result does not match original position {index}")
        response = saved["response"]["choices"][0]["message"]["content"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"No final answer at original position {index}")
        results.append(longvideobench_process_results(original, [response])["lvb_acc"])
    print(
        json.dumps(
            {
                "fixed_positions": [0, 1, 2],
                "seed": args.seed,
                "lvb_acc": longvideobench_aggregate_results(results),
            }
        )
    )


if __name__ == "__main__":
    main()
