# SPDX-License-Identifier: Apache-2.0
"""Reproduce and verify the external inputs used by issue 135."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
from pathlib import Path

REPORT_DIR = Path("reports/issue_135_readiness_slo_matrix")
BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
BURSTGPT_RAW_SHA256 = "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"  # pragma: allowlist secret
BURSTGPT_PROCESSED_SHA256 = "ef3bc195a041df6e35fd2f0572b93ed0c393482d3ec91b35e46c75bc409f6104"  # pragma: allowlist secret
MODEL_INFO_SHA256 = "87b61d3fbe4ccbceb12774955b252b11d1065437cd5edc57edf2058dd2f5f644"  # pragma: allowlist secret
MODEL_INFO = REPORT_DIR / (
    "input-provenance/vllm-model_executor-models-qwen2-Qwen2ForCausalLM.json"
)
OUTPUT_FIELDS = (
    "Timestamp",
    "Session ID",
    "Request tokens",
    "Response tokens",
    "Model",
    "Total tokens",
    "Log Type",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def require_model_info_hash(path: Path) -> None:
    # The cached artifact had no final newline. The committed JSON follows the
    # repository text-file convention, so authenticate the original bytes after
    # removing exactly that repository-added newline.
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise SystemExit(f"expected repository newline in {path}")
    actual = hashlib.sha256(data[:-1]).hexdigest()
    if actual != MODEL_INFO_SHA256:
        raise SystemExit(
            f"SHA-256 mismatch for recorded model-info bytes: "
            f"expected {MODEL_INFO_SHA256}, got {actual}"
        )


def download_raw(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(BURSTGPT_URL) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, length=1024 * 1024)
        require_hash(partial, BURSTGPT_RAW_SHA256)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def preprocess(raw: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        raw.open(newline="", encoding="utf-8-sig") as source,
        destination.open("w", newline="", encoding="utf-8") as output,
    ):
        reader = csv.DictReader(source)
        missing = set(OUTPUT_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"BurstGPT input is missing columns: {sorted(missing)}")
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if row["Model"] == "GPT-4" and int(row["Response tokens"]) > 0:
                writer.writerow({field: row[field] for field in OUTPUT_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, help="reuse an existing BurstGPT_3.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR / "inputs",
        help="generated input directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = args.raw_csv or args.output_dir / "BurstGPT_3.csv"
    if args.raw_csv is None and not raw.exists():
        download_raw(raw)
    require_hash(raw, BURSTGPT_RAW_SHA256)

    processed = args.output_dir / "BurstGPT_processed.csv"
    preprocess(raw, processed)
    require_hash(processed, BURSTGPT_PROCESSED_SHA256)
    require_model_info_hash(MODEL_INFO)
    print(f"verified raw BurstGPT input: {raw}")
    print(f"verified processed BurstGPT input: {processed}")
    print(f"verified model-info evidence: {MODEL_INFO}")


if __name__ == "__main__":
    main()
