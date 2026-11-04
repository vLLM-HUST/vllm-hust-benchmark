"""Check fixed LongVideoBench validation inputs through the upstream loader."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-loader", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--position", type=int, choices=range(3), action="append")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("longvideobench_author", args.author_loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.LongVideoBenchDataset(str(args.data_root), "lvb_val.json", max_num_frames=256)
    if len(dataset) != 1337:
        raise ValueError("Expected the pinned official validation split (1337 rows)")
    for index in args.position or range(3):
        row = dataset[index]
        inputs = row["inputs"]
        frame_count = sum(not isinstance(item, str) for item in inputs)
        if not 0 < frame_count <= 256 or row["correct_choice"] not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"Malformed upstream input at position {index}")
        print(json.dumps({
            "position": index, "frames": frame_count,
            "text_segments": len(inputs) - frame_count,
            "reference_kept_separate": True,
        }), flush=True)


if __name__ == "__main__":
    main()
