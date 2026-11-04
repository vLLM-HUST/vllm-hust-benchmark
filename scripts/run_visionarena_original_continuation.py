"""Validate or run the fixed first three VisionArena original continuations."""
import hashlib
import io
import json
import argparse
import base64
import urllib.request
from pathlib import Path

def continuation_prefix(messages, raw):
    prefix = [dict(message) for message in messages[:-1]]
    prefix[0]["content"] = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
            base64.b64encode(raw).decode("ascii")}},
        {"type": "text", "text": prefix[0]["content"]},
    ]
    return prefix


def main():
    from PIL import Image
    from pyarrow.parquet import ParquetFile

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.endpoint and not (args.model and args.max_tokens and args.output):
        parser.error("execution requires explicit model, generation budget and output directory")
    if args.endpoint:
        args.output.mkdir(mode=0o700)
    shard = args.shard
    batch = next(ParquetFile(shard).iter_batches(
        batch_size=3, columns=["images", "conversation"]
    ))
    receipt = []
    for position, row in enumerate(batch.to_pylist()):
        messages = [message for group in row["conversation"] for message in group]
        assert [message["role"] for message in messages] == [
            role for _ in range(len(messages) // 2) for role in ("user", "assistant")
        ]
        assert len(row["images"]) == 1
        raw = row["images"][0]["bytes"]
        with Image.open(io.BytesIO(raw)) as image:
            size, fmt = image.size, image.format
            image.verify()
        if fmt != "PNG":
            raise ValueError(f"Expected original PNG at fixed position {position}")
        if args.endpoint:
            prefix = continuation_prefix(messages, raw)
            payload = {"model": args.model, "messages": prefix, "max_tokens": args.max_tokens}
            request = urllib.request.Request(
                args.endpoint.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
            (args.output / f"{position}.json").write_text(json.dumps(result), encoding="utf-8")
            choice = result["choices"][0]
            if not choice["message"]["content"]:
                raise RuntimeError(f"Empty output for fixed position {position}")
        receipt.append({
            "position": position, "prefix_messages": len(messages) - 1,
            "reference_kept_separate": True,
            "image_sha256": hashlib.sha256(raw).hexdigest(),
            "image_size": size, "image_format": fmt,
        })
    assert len(receipt) == 3
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
