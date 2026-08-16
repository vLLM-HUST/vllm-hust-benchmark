#!/usr/bin/env python3
"""Prepare the minimal offline VisionArena input for the official seed-0 run.

The official v0.18.0 workload streams VisionArena-Chat, shuffles it with seed 0
and the Hugging Face datasets 3.3.0 default 1,000-row buffer, then takes 1,000
requests.  For the pinned 43-shard dataset, shard 40 is first after the shard
shuffle and contains enough rows to supply the entire shuffle buffer and all
1,000 replacements.  A one-shard local dataset is therefore byte-for-byte
equivalent for the selected requests, without downloading the other 42 shards.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

REPO_ID = "lmarena-ai/VisionArena-Chat"
REVISION = "1394b4f59ab6f1f2e5aff6bc15b448e15960e170"  # pragma: allowlist secret
SOURCE_RELATIVE_PATH = "data/train-00040-of-00043.parquet"
FROZEN_RELATIVE_PATH = "data/train-00000-of-00001.parquet"
EXPECTED_SHARD_COUNT = 43
EXPECTED_SHARD_ROWS = 4628
EXPECTED_SHARD_SIZE = 1_942_220_201
EXPECTED_SHARD_SHA256 = "1433e9c328f817c4176b50ab080f9c85ae27c5c6060f546e020e384c1acb92c5"  # pragma: allowlist secret
REFERENCE_DATASETS_VERSION = "3.3.0"
SEED = 0
SHUFFLE_BUFFER_SIZE = 1000
NUM_PROMPTS = 1000
EXPECTED_SHARD_ORDER_SHA256 = "c2121f82122d83bc005730dab935defc9175af24013d10b868d97b64917306b2"  # pragma: allowlist secret
EXPECTED_SOURCE_INDEX_SHA256 = "6e1b1d169e836663328c6f9a38145ed093192f93f36656b26327f2244eb583bc"  # pragma: allowlist secret
EXPECTED_CONTENT_SHA256 = "2b41a850b78bc901caedf7e4d86ce52fc1804edc584f5f4da53a070df2a34b41"  # pragma: allowlist secret


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_selection() -> tuple[list[int], list[int]]:
    """Return the canonical shard order and selected row indices.

    This mirrors datasets 3.3.0 ``IterableDataset.shuffle``: the shard list and
    row buffer each receive a deepcopy of the same seed-0 generator, so their
    random streams both start from seed 0.
    """

    import numpy as np

    shard_order = list(range(EXPECTED_SHARD_COUNT))
    np.random.default_rng(SEED).shuffle(shard_order)

    row_rng = np.random.default_rng(SEED)
    buffer = list(range(SHUFFLE_BUFFER_SIZE))
    selected: list[int] = []
    random_slots = row_rng.integers(0, SHUFFLE_BUFFER_SIZE, size=NUM_PROMPTS)
    for source_row, slot in zip(
        range(SHUFFLE_BUFFER_SIZE, SHUFFLE_BUFFER_SIZE + NUM_PROMPTS),
        random_slots,
        strict=True,
    ):
        index = int(slot)
        selected.append(buffer[index])
        buffer[index] = source_row
    return shard_order, selected


def _framed_row_digest(item: dict[str, Any]) -> bytes:
    images = item.get("images") or []
    image = (images[0].get("bytes") or b"") if images else b""
    conversation = item.get("conversation") or []
    prompt = conversation[0][0].get("content") if conversation else ""
    fields = (
        str(item.get("conversation_id") or "").encode("utf-8"),
        str(prompt or "").encode("utf-8"),
        image,
    )
    digest = hashlib.sha256()
    for field in fields:
        digest.update(struct.pack(">Q", len(field)))
        digest.update(field)
    return digest.digest()


def selected_content_sha256(shard: Path, selected: list[int]) -> str:
    """Hash the ordered selected request identities without retaining images."""

    import pyarrow.parquet as pq

    needed = set(selected)
    maximum = max(needed)
    row_digests: dict[int, bytes] = {}
    row_index = 0
    parquet = pq.ParquetFile(shard)
    for batch in parquet.iter_batches(
        batch_size=16,
        columns=["images", "conversation_id", "conversation"],
    ):
        for item in batch.to_pylist():
            if row_index in needed:
                row_digests[row_index] = _framed_row_digest(item)
            row_index += 1
            if row_index > maximum:
                break
        if row_index > maximum:
            break
    if row_digests.keys() != needed:
        missing = sorted(needed - row_digests.keys())
        raise ValueError(f"source shard is missing selected rows: {missing[:10]}")
    return hashlib.sha256(
        b"".join(row_digests[index] for index in selected)
    ).hexdigest()


def verify_source(cache_root: Path) -> tuple[Path, dict[str, Any]]:
    import pyarrow.parquet as pq

    repo_root = cache_root / "datasets--lmarena-ai--VisionArena-Chat"
    tree_path = repo_root / "trees" / f"{REVISION}.json"
    snapshot = repo_root / "snapshots" / REVISION
    shard = snapshot / SOURCE_RELATIVE_PATH
    if not tree_path.is_file():
        raise FileNotFoundError(f"missing pinned tree manifest: {tree_path}")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    parquet_entries = {
        name: entry
        for name, entry in tree.get("files", {}).items()
        if name.endswith(".parquet")
    }
    if len(parquet_entries) != EXPECTED_SHARD_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SHARD_COUNT} parquet shards in manifest, "
            f"found {len(parquet_entries)}"
        )
    entry = parquet_entries.get(SOURCE_RELATIVE_PATH)
    if entry is None:
        raise ValueError(f"pinned tree does not contain {SOURCE_RELATIVE_PATH}")
    if int(entry.get("lfs_size", -1)) != EXPECTED_SHARD_SIZE:
        raise ValueError(
            "shard size in tree manifest does not match the audit contract"
        )
    if entry.get("lfs_sha256") != EXPECTED_SHARD_SHA256:
        raise ValueError(
            "shard digest in tree manifest does not match the audit contract"
        )
    if not shard.is_file():
        raise FileNotFoundError(f"missing canonical shard: {shard}")
    if shard.stat().st_size != EXPECTED_SHARD_SIZE:
        raise ValueError(f"unexpected shard size: {shard.stat().st_size}")
    actual_sha256 = _sha256_file(shard)
    if actual_sha256 != EXPECTED_SHARD_SHA256:
        raise ValueError(f"unexpected shard SHA-256: {actual_sha256}")
    parquet = pq.ParquetFile(shard)
    if parquet.metadata.num_rows != EXPECTED_SHARD_ROWS:
        raise ValueError(f"unexpected shard row count: {parquet.metadata.num_rows}")
    return shard, tree


def audit(cache_root: Path) -> dict[str, Any]:
    shard, _tree = verify_source(cache_root)
    shard_order, selected = derive_selection()
    if shard_order[0] != 40:
        raise ValueError(f"seed-0 canonical first shard changed: {shard_order[0]}")
    shard_order_sha256 = _compact_json_sha256(shard_order)
    source_index_sha256 = _compact_json_sha256(selected)
    content_sha256 = selected_content_sha256(shard, selected)
    expected = {
        "shard_order_sha256": EXPECTED_SHARD_ORDER_SHA256,
        "source_index_sha256": EXPECTED_SOURCE_INDEX_SHA256,
        "content_sha256": EXPECTED_CONTENT_SHA256,
    }
    actual = {
        "shard_order_sha256": shard_order_sha256,
        "source_index_sha256": source_index_sha256,
        "content_sha256": content_sha256,
    }
    if actual != expected:
        raise ValueError(
            f"frozen selection audit mismatch: expected={expected}, actual={actual}"
        )
    return {
        "schema_version": "visionarena-frozen-input/v1",
        "dataset": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "split": "train",
            "canonical_shard_count": EXPECTED_SHARD_COUNT,
        },
        "source_shard": {
            "canonical_index": 40,
            "relative_path": SOURCE_RELATIVE_PATH,
            "rows": EXPECTED_SHARD_ROWS,
            "size_bytes": EXPECTED_SHARD_SIZE,
            "sha256": EXPECTED_SHARD_SHA256,
        },
        "reference_loader": {
            "library": "datasets",
            "version": REFERENCE_DATASETS_VERSION,
            "streaming": True,
            "shuffle_seed": SEED,
            "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
            "num_prompts": NUM_PROMPTS,
        },
        "selection": {
            "canonical_first_shard": 40,
            "source_rows_consumed": SHUFFLE_BUFFER_SIZE + NUM_PROMPTS,
            "maximum_selected_source_row": max(selected),
            "selected_rows": len(selected),
            "shard_order_sha256": shard_order_sha256,
            "source_index_encoding": "compact-json-array-of-zero-based-row-indices",
            "source_index_sha256": source_index_sha256,
            "content_digest_encoding": (
                "sha256(concat(sha256(u64be-len+conversation_id-utf8,"
                "u64be-len+prompt-utf8,u64be-len+first-image-bytes)))"
            ),
            "content_sha256": content_sha256,
        },
        "frozen_layout": {
            "dataset_relative_path": FROZEN_RELATIVE_PATH,
            "logical_hf_name": REPO_ID,
        },
    }


def prepare(output_dir: Path, shard: Path) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / FROZEN_RELATIVE_PATH
    if os.path.lexists(destination):
        if not destination.is_symlink() or destination.resolve() != shard.resolve():
            raise FileExistsError(
                f"refusing to replace non-matching frozen input: {destination}"
            )
        return destination
    relative_target = os.path.relpath(shard, destination.parent)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.symlink_to(relative_target)
    os.replace(temporary, destination)
    return destination


def validate_local_loader(output_dir: Path) -> str:
    """Exercise the same local streaming path used by the benchmark CLI."""

    from datasets import load_dataset

    with tempfile.TemporaryDirectory(prefix="visionarena-loader-audit-") as cache_dir:
        dataset = load_dataset(
            str(output_dir), split="train", streaming=True, cache_dir=cache_dir
        )
        dataset = dataset.shuffle(seed=SEED)
        digest = hashlib.sha256()
        count = 0
        for item in dataset:
            digest.update(_framed_row_digest(item))
            count += 1
            if count == NUM_PROMPTS:
                break
    if count != NUM_PROMPTS:
        raise ValueError(f"local frozen input yielded only {count} requests")
    actual = digest.hexdigest()
    if actual != EXPECTED_CONTENT_SHA256:
        raise ValueError(
            "local datasets loader changed the selected content: "
            f"expected={EXPECTED_CONTENT_SHA256}, actual={actual}"
        )
    return importlib.metadata.version("datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/shared_datasets/vllm-hust-benchmark/huggingface/hub"),
        help="Hugging Face hub cache root containing the pinned dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/shared_datasets/vllm-hust-benchmark/huggingface/frozen-inputs/"
            "VisionArena-Chat-1394b4f-seed0-1000"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="runtime audit receipt (default: <output-dir>.manifest.json)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="audit without creating the local view",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = audit(args.cache_root)
    shard = (
        args.cache_root
        / "datasets--lmarena-ai--VisionArena-Chat"
        / "snapshots"
        / REVISION
        / SOURCE_RELATIVE_PATH
    )
    if not args.verify_only:
        destination = prepare(args.output_dir, shard)
        datasets_version = validate_local_loader(args.output_dir)
        receipt = args.receipt or args.output_dir.with_suffix(".manifest.json")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        runtime_manifest = {
            **manifest,
            "runtime": {
                "source_shard": str(shard.resolve()),
                "frozen_dataset_path": str(args.output_dir.resolve()),
                "frozen_parquet_path": str(destination),
                "validated_datasets_version": datasets_version,
            },
        }
        temporary = receipt.with_name(f".{receipt.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, receipt)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
