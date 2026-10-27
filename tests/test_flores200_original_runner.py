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
import score_flores200_original_results as scorer  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "flores_runner", ROOT / "scripts/run_flores200_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path, prefix=""):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for split in ("dev", "devtest"):
            for code in ("eng_Latn", "fra_Latn"):
                raw = ("first fixture\n\nthird fixture\nfourth fixture\n").encode()
                info = tarfile.TarInfo(
                    f"{prefix}flores200_dataset/{split}/{code}.{split}"
                )
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


@pytest.mark.parametrize("corruption", [None, "reference", "order", "prediction"])
def test_saved_native_scoring_keeps_original_pairs(tmp_path, corruption):
    archive, dataset = fixture(tmp_path)
    original = runner.original_pairs(
        archive, dataset, source="eng_Latn", target="fra_Latn", split="devtest"
    )
    rows = [{**row, "prediction": "fixture prediction"} for row in original]
    if corruption == "reference":
        rows[0]["reference"] = "changed reference"
    elif corruption == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif corruption == "prediction":
        rows[0]["prediction"] = None
    path = tmp_path / "saved.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    kwargs = dict(
        source="eng_Latn", target="fra_Latn", split="devtest", dataset=dataset
    )
    if corruption:
        with pytest.raises(ValueError):
            scorer.load_saved(path, archive, **kwargs)
    else:
        assert scorer.load_saved(path, archive, **kwargs) == rows


def test_secondary_metric_encodes_both_sides_without_extra_tokenization():
    class Metric:
        def __init__(self):
            self.calls = []

        def corpus_score(self, hypotheses, references):
            self.calls.append((hypotheses, references))
            return NS(score=12.0)

        def get_signature(self):
            return "inert-fixture-not-a-real-metric"

    primary, secondary = Metric(), Metric()
    rows = [{"prediction": "pred", "reference": "ref"}]
    proof = scorer.calculate(
        rows, chrf=primary, bleu=secondary, pieces=lambda text: ["piece", text]
    )
    assert primary.calls == [(["pred"], [["ref"]])]
    assert secondary.calls == [(["piece pred"], [["piece ref"]])]
    assert proof["spbleu"] == 12.0
    assert "tokenize=none" in proof["spbleu_tokenization"]


def test_frozen_delivery_links_native_scoring_installation_and_consumer_evidence():
    """Receipt integrity only; this fixture does not repeat model execution."""
    receipt_path = (
        ROOT / "docs/workload-data/validation/flores200-native-delivery-20261026.json"
    )
    proof = json.loads(receipt_path.read_text("utf-8"))
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    entry = ledger["delivery_inventory"]["FLORES200"]
    assert (
        entry["checked"]
        and entry["delivery_receipt"] == receipt_path.relative_to(ROOT).as_posix()
    )
    assert proof["source_checkbox_eligible"]
    assert proof["native"]["translations_reused"] == 9
    assert (
        sum(
            len(profile["original_positions"])
            for profile in proof["native"]["profiles"]
        )
        == 9
    )
    assert proof["fixed_consumer"]["noneditable_installed_payload_files_equal"] == 42
    assert proof["fixed_consumer"]["native_results_public_and_research_roundtrip"] == 9
    assert not proof["fixed_consumer"]["research_default_dependency_updated"]
    assert (
        proof["client"]["isolated_pip_check_with_empty_pythonpath"]
        == "No broken requirements found."
    )
    assert proof["model_calls_for_closure"] == 0
    assert not proof["raw_dataset_predictions_or_model_published"]
    for path in proof["native"]["historical_receipts_reused"]:
        assert (ROOT / path).is_file()


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


def test_real_audit_does_not_claim_full_source_or_extra_unique_rows():
    audit = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/flores200-two-profile-native-audit-20261025.json"
        ).read_text("utf-8")
    )
    assert audit["native_model_calls"] == 6
    assert audit["unique_original_parallel_row_positions"] == 3
    assert audit["translation_direction_profiles"] == 2
    assert audit["model_calls_by_audit"] == 0
    assert all(
        row["original_indices"] == [0, 1, 2]
        and row["all_original_source_and_reference_rows_equal"]
        for row in audit["profiles"]
    )
    assert not audit["all_languages_or_directions_native_validated"]
    assert not audit["source_checkbox_eligible"]
    assert len(audit["unsupported_model_language_codes"]) == 3


def test_original_archive_identity_required(tmp_path):
    archive, dataset = fixture(tmp_path)
    dataset["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.original_pairs(
            archive, dataset, source="eng_Latn", target="fra_Latn", split="devtest"
        )


def test_original_dot_prefixed_tar_names_resolve_without_rewriting(tmp_path):
    archive, dataset = fixture(tmp_path, prefix="./")
    selected = runner.original_pairs(
        archive, dataset, source="eng_Latn", target="fra_Latn", split="devtest"
    )
    assert [row["row_index"] for row in selected] == [0, 1, 2]
    assert selected[0]["source"] == "first fixture"


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
