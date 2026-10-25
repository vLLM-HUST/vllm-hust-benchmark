"""Tiny fabricated archive and transport fixtures; no publisher/model run."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "flores_runner", ROOT / "scripts/run_flores200_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for split in ("dev", "devtest"):
            for code in ("eng_Latn", "fra_Latn"):
                raw = ("first fixture\n\nthird fixture\nfourth fixture\n").encode()
                info = tarfile.TarInfo(f"flores200_dataset/{split}/{code}.{split}")
                info.size = len(raw)
                bundle.addfile(info, io.BytesIO(raw))
    raw = archive.read_bytes()
    dataset = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "split_rows": {"dev": 4, "devtest": 4},
        "language_code_count": 2,
    }
    return archive, dataset


def pairs(tmp_path):
    archive, dataset = fixture(tmp_path)
    return runner.original_pairs(
        archive, dataset, source="eng_Latn", target="fra_Latn", split="devtest"
    )


def test_original_blank_row_alignment_and_order_preserved(tmp_path):
    selected = pairs(tmp_path)
    assert [pair["row_index"] for pair in selected] == [0, 1, 2]
    assert selected[1]["source"] == selected[1]["reference"] == ""
    archive, dataset = fixture(tmp_path)
    assert (
        len(
            runner.original_pairs(
                archive,
                dataset,
                source="eng_Latn",
                target="fra_Latn",
                split="devtest",
                full=True,
            )
        )
        == 4
    )


def test_original_archive_identity_required(tmp_path):
    archive, dataset = fixture(tmp_path)
    dataset["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.original_pairs(
            archive, dataset, source="eng_Latn", target="fra_Latn", split="devtest"
        )


def test_unmodified_sentences_reach_translator_and_references_only_scorer(tmp_path):
    selected = pairs(tmp_path)
    seen = []

    def translate(text):
        seen.append(text)
        return "fixture prediction"

    def score(hypotheses, references):
        assert len(hypotheses) == 3
        assert references == [[pair["reference"] for pair in selected]]
        return NS(score=0.0)

    result = runner.execute(
        selected,
        translate=translate,
        metric=NS(corpus_score=score, get_signature=lambda: "fixture"),
        output=tmp_path / "out",
        source="eng_Latn",
        target="fra_Latn",
        split="devtest",
    )
    assert seen == [pair["source"] for pair in selected]
    assert result["native_execution_complete"]
    assert not result["source_checkbox_eligible"]


def test_failure_preserves_partial_positions_without_replacement(tmp_path):
    seen = []

    def translate(text):
        seen.append(text)
        if len(seen) == 2:
            raise RuntimeError("fixture resource failure")
        return "fixture prediction"

    with pytest.raises(RuntimeError):
        runner.execute(
            pairs(tmp_path),
            translate=translate,
            metric=None,
            output=tmp_path / "out",
            source="eng_Latn",
            target="fra_Latn",
            split="devtest",
        )
    receipt = json.loads((tmp_path / "out/receipt-private.json").read_text())
    assert receipt["completed_original_positions"] == [0]
    assert not receipt["native_execution_complete"]
