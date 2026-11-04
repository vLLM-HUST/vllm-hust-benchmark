"""Check fixed LongVideoBench validation inputs through the upstream loader."""
import argparse
import base64
import importlib.util
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen


def interleaved_content(inputs):
    for item in inputs:
        if isinstance(item, str):
            yield {"type": "text", "text": item}
        else:
            with io.BytesIO() as buffer:
                item.save(buffer, format="PNG")
                image = base64.b64encode(buffer.getvalue()).decode("ascii")
            yield {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-loader", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--position", type=int, choices=range(3), action="append")
    parser.add_argument("--endpoint", help="OpenAI-compatible /v1 URL; omit for CPU preflight")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output", type=Path, help="New private response directory")
    args = parser.parse_args()
    if args.endpoint:
        if not (args.model and args.max_tokens and args.max_tokens > 0 and args.output):
            parser.error("execution requires model, positive output budget, and private output directory")
        args.output.mkdir(mode=0o700)
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
        if args.endpoint:
            request = Request(
                args.endpoint.rstrip("/") + "/chat/completions",
                data=json.dumps({
                    "model": args.model, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": list(interleaved_content(inputs))}],
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=1200) as response:
                payload = json.load(response)
            (args.output / f"{index}.json").write_text(
                json.dumps({"id": row["id"], "reference": row["correct_choice"], "response": payload}),
                encoding="utf-8",
            )
        print(json.dumps({
            "position": index, "frames": frame_count,
            "text_segments": len(inputs) - frame_count,
            "reference_kept_separate": True,
        }), flush=True)


if __name__ == "__main__":
    main()
