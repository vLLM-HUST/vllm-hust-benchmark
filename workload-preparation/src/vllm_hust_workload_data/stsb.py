"""Original SBERT-published STSBenchmark TSV, without chat conversion."""

import csv
import gzip
import math
from pathlib import Path


FIELDS = ["split", "genre", "dataset", "year", "sid", "score", "sentence1", "sentence2"]
SPLITS = {"train", "dev", "test"}


def original_rows(path):
    """Keep all original strings; normalize scores only inside the native evaluator."""
    path = Path(path)
    if path.name != "stsbenchmark.tsv.gz":
        raise ValueError("original SBERT STSBenchmark gzip filename required")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        # This matches the publisher's training_stsbenchmark.py loader. Quotes
        # in the original sentences are content, not a CSV quoting protocol.
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != FIELDS:
            raise ValueError("original STSBenchmark column order required")
        for row in reader:
            if None in row or any(not isinstance(value, str) for value in row.values()):
                raise ValueError("original STSBenchmark field width mismatch")
            if (
                row["split"] not in SPLITS
                or not row["sentence1"]
                or not row["sentence2"]
            ):
                raise ValueError(
                    "original STSBenchmark split and sentence pair required"
                )
            try:
                score = float(row["score"])
            except ValueError:
                raise ValueError(
                    "original continuous similarity score required"
                ) from None
            if not math.isfinite(score) or not 0 <= score <= 5:
                raise ValueError(
                    "original similarity score must be finite and within 0..5"
                )
            yield row
