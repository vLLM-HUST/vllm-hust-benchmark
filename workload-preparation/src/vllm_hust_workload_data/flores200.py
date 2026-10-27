"""Original FLORES parallel assets and explicit directional field projections.

No tokenizer, model, network, scoring, language substitution or research analysis.
The bounded tar checks preserve the existing original-asset preparer's protocol.
"""

import csv
from copy import deepcopy
import io
from pathlib import Path, PurePosixPath
import re
import tarfile

from .assets import sha256
from .mapping_contract import validate_mapping
from .registry import source_spec

LANGUAGE = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}")
METADATA_FIELDS = ("URL", "domain", "topic", "has_image", "has_hyperlink")


def dataset_spec(*, catalog_root=None):
    source = source_spec("FLORES200", catalog_root)
    files = source["files"]
    if len(files) != 1 or files[0]["path"] != "flores200_dataset.tar.gz":
        raise ValueError("explicit original Meta archive binding required")
    return {**files[0], **source["archive_layout"]}


def parallel_rows(path, *, dataset=None):
    """Original combined dev/devtest row locations with all aligned languages.

    This iterator yields corpus rows, not ready model prompts. Choose an explicit
    translation_mapping to keep the target/metadata outside the model input.
    Files stay in their original archive; no extraction or synthetic data.
    """
    path = Path(path)
    dataset = dataset_spec() if dataset is None else dataset

    def verify():
        if path.stat().st_size != dataset["bytes"] or sha256(path) != dataset["sha256"]:
            raise ValueError("original FLORES archive identity mismatch")

    verify()
    parallel = {split: {} for split in dataset["split_rows"]}
    metadata = {}
    seen, total = set(), 0
    with tarfile.open(path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("unsafe original FLORES archive member")
            if member.isdir():
                continue
            normalized = str(name)
            if normalized in seen or not normalized.startswith("flores200_dataset/"):
                raise ValueError("duplicate or unexpected original archive member")
            seen.add(normalized)
            total += member.size
            if member.size > 4 * 1024**2 or total > 256 * 1024**2 or len(seen) > 2048:
                raise ValueError("archive exceeds original inspected safety bounds")
            raw = archive.extractfile(member).read()
            if len(raw) != member.size:
                raise ValueError("incomplete original archive member")
            text = raw.decode("utf-8")
            match = re.fullmatch(
                r"flores200_dataset/(dev|devtest)/([a-z]{3}_[A-Z][a-z]{3})\.(dev|devtest)",
                normalized,
            )
            if match:
                split, code, suffix = match.groups()
                if split != suffix or split not in parallel:
                    raise ValueError("original split/member mismatch")
                lines = text.split("\n")
                if lines[-1] == "":
                    lines.pop()
                if len(lines) != dataset["split_rows"][split]:
                    raise ValueError("original parallel line count mismatch")
                parallel[split][code] = lines
            else:
                match = re.fullmatch(
                    r"flores200_dataset/metadata_(dev|devtest)\.tsv", normalized
                )
                if match:
                    split = match.group(1)
                    reader = csv.DictReader(
                        io.StringIO(text, newline=""), delimiter="\t"
                    )
                    if tuple(reader.fieldnames or ()) != METADATA_FIELDS:
                        raise ValueError("original FLORES metadata fields differ")
                    rows = list(reader)
                    if len(rows) != dataset["split_rows"][split] or any(
                        None in row or None in row.values() for row in rows
                    ):
                        raise ValueError(
                            "original metadata/parallel row alignment mismatch"
                        )
                    metadata[split] = rows
    language_sets = [set(values) for values in parallel.values()]
    if any(len(codes) != dataset["language_code_count"] for codes in language_sets):
        raise ValueError("complete original language coverage required")
    if any(codes != language_sets[0] for codes in language_sets) or set(
        metadata
    ) != set(parallel):
        raise ValueError("original language/split metadata alignment mismatch")
    codes = sorted(language_sets[0])
    # Stable archive-row locator order, not a serving request arrival order.
    for split in sorted(parallel):
        for index in range(dataset["split_rows"][split]):
            yield {
                "split": split,
                "row_index": index,
                "parallel_texts": {
                    code: parallel[split][code][index] for code in codes
                },
                "language_codes": {code: code for code in codes},
                "publisher_metadata": deepcopy(metadata[split][index]),
            }
    verify()


def translation_mapping(source_language, target_language, *, catalog_root=None):
    """An explicit data-only direction, not a new prompt or dataset variant.

    Language strings come unchanged from original member names. Identity remains
    the archive row location; the profile hash distinguishes directional views.
    """
    if source_language == target_language or any(
        not isinstance(code, str) or not LANGUAGE.fullmatch(code)
        for code in (source_language, target_language)
    ):
        raise ValueError("distinct unchanged original language/script codes required")
    source = source_spec("FLORES200", catalog_root)

    def text(path):
        return {"path": path, "type": "string", "required": True}

    mapping = {
        "mapping_version": "source-mapping/v1",
        "source": {
            "source_id": "FLORES200",
            "revision": source["revision"],
            "unit": "original_parallel_row",
            "publisher": "Meta FLORES authors",
            "locator": source["source_url"],
            "license": "CC-BY-SA-4.0; raw assets external",
            "provenance_kind": "official_parallel_corpus_directional_projection",
        },
        "families": ["text-transformation"],
        "records_path": [],
        "identity": {"task_id": None},
        "input": {
            "source_text": text(["parallel_texts", source_language]),
            "source_language": text(["language_codes", source_language]),
            "target_language": text(["language_codes", target_language]),
            "split": text(["split"]),
            "row_index": {"path": ["row_index"], "type": "number", "required": True},
        },
        "features": {},
        "outcome": {
            "reference_text": text(["parallel_texts", target_language]),
            "publisher_metadata": {
                "path": ["publisher_metadata"],
                "type": "object",
                "required": True,
                "fields": {name: text([name]) for name in METADATA_FIELDS},
            },
        },
        "runtime": {
            "kind": "task_dataset",
            "requirements": [
                "Original source/target language codes and split/line coordinates; no language substitution.",
                "Only source_text plus explicit generation direction is model input; reference and metadata remain separate.",
            ],
            "environment_ref": "scripts/run_flores200_original.py",
            "termination_ref": "original model generation configuration",
            "scorer_ref": "FLORES author chrF++ and spBLEU protocols",
        },
        "replay": {
            "order": "source_file_order_not_assumed_arrival_order",
            "artificial_load": False,
        },
        "evidence": {
            "field_reference": "Original official Meta archive: per-language dev/devtest physical lines and aligned metadata TSV; explicit source/target projection only.",
            "review_status": "field_contract_reviewed",
        },
    }
    return validate_mapping(mapping)
