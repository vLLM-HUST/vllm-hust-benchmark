#!/usr/bin/env python3
"""Bind native evidence fields to the currently running frozen test."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Normalized evidence fields from the native exporter",
    )
    args = parser.parse_args()
    fields = json.loads(args.data.read_text(encoding="utf-8-sig"))
    identity = {
        "case": os.environ["VLLM_HUST_TEST_CASE"],
        "arm": os.environ["VLLM_HUST_TEST_ARM"],
        "repeat": int(os.environ["VLLM_HUST_TEST_REPEAT"]),
        "contract_sha256": digest(os.environ["VLLM_HUST_TEST_CONTRACT"]),
        "config_sha256": digest(os.environ["VLLM_HUST_TEST_CONFIG"]),
    }
    if fields.keys() & identity.keys():
        raise ValueError("Normalized data must not supply runner identity fields")
    receipt = Path(os.environ["VLLM_HUST_TEST_OUTPUT"]) / "receipt.json"
    with receipt.open("x", encoding="utf-8") as file:
        json.dump(
            {**identity, **fields}, file, ensure_ascii=False, allow_nan=False, indent=2
        )
        file.write("\n")


if __name__ == "__main__":
    main()
