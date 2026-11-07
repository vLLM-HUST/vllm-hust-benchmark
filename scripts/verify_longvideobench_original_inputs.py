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
            yield {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"},
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-loader", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--position", type=int, choices=range(3), action="append")
    parser.add_argument(
        "--endpoint", help="OpenAI-compatible /v1 URL; omit for CPU preflight"
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--capacity-model", type=Path, help="Local processor for CPU token accounting"
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output", type=Path, help="New private response directory")
    args = parser.parse_args()
    processor = None
    if args.capacity_model:
        if args.endpoint:
            parser.error("capacity preflight does not send model requests")
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            args.capacity_model, local_files_only=True
        )
    if args.endpoint:
        if not (args.model and args.max_tokens and args.max_tokens > 0 and args.output):
            parser.error(
                "execution requires model, positive output budget, and private output directory"
            )
        args.output.mkdir(mode=0o700)
    spec = importlib.util.spec_from_file_location(
        "longvideobench_author", args.author_loader
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.LongVideoBenchDataset(
        str(args.data_root), "lvb_val.json", max_num_frames=256
    )
    if len(dataset) != 1337:
        raise ValueError("Expected the pinned official validation split (1337 rows)")
    for index in args.position or range(3):
        row = dataset[index]
        inputs = row["inputs"]
        frame_count = sum(not isinstance(item, str) for item in inputs)
        frame_sizes = sorted(
            {item.size for item in inputs if not isinstance(item, str)}
        )
        if not 0 < frame_count <= 256 or row["correct_choice"] not in {
            "A",
            "B",
            "C",
            "D",
            "E",
        }:
            raise ValueError(f"Malformed upstream input at position {index}")
        if processor is not None:
            parts = []
            grids = {}
            visual = 0
            for item in inputs:
                if isinstance(item, str):
                    parts.append({"type": "text", "text": item})
                else:
                    parts.append({"type": "image"})
                    if item.size not in grids:
                        grid = processor.image_processor(
                            images=[item], return_tensors="np"
                        )["image_grid_thw"][0]
                        grids[item.size] = (
                            int(grid.prod()) // processor.image_processor.merge_size**2
                        )
                    visual += grids[item.size]
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": parts}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if rendered.count(processor.image_token) != frame_count:
                raise ValueError(
                    "Processor placeholder count differs from original frames"
                )
            text = len(processor.tokenizer.encode(rendered, add_special_tokens=False))
            print(
                json.dumps(
                    {
                        "position": index,
                        "prompt_tokens": text - frame_count + visual,
                        "visual_tokens": visual,
                        "capacity_probe_only": True,
                    }
                ),
                flush=True,
            )
        if args.endpoint:
            request = Request(
                args.endpoint.rstrip("/") + "/chat/completions",
                data=json.dumps(
                    {
                        "model": args.model,
                        "max_tokens": args.max_tokens,
                        "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "messages": [
                            {
                                "role": "user",
                                "content": list(interleaved_content(inputs)),
                            }
                        ],
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=1200) as response:
                payload = json.load(response)
            (args.output / f"{index}.json").write_text(
                json.dumps(
                    {
                        "id": row["id"],
                        "reference": row["correct_choice"],
                        "response": payload,
                    }
                ),
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "position": index,
                    "frames": frame_count,
                    "frame_sizes": frame_sizes,
                    "text_segments": len(inputs) - frame_count,
                    "reference_kept_separate": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
