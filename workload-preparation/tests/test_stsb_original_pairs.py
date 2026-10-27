import gzip

import pytest

from vllm_hust_workload_data.stsb import FIELDS, original_rows
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec


def write_pair(
    tmp_path, *, split="test", score="4.500", sentence='"Quoted original"', extra=""
):
    path = tmp_path / "stsbenchmark.tsv.gz"
    values = [split, "news", "source", "2017", "42", score, sentence, "Other sentence"]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(FIELDS) + "\n" + "\t".join(values) + extra + "\n")
    return path


def test_original_strings_and_quotes_are_not_rewritten(tmp_path):
    (row,) = original_rows(write_pair(tmp_path))
    assert row["sentence1"] == '"Quoted original"'
    assert row["score"] == "4.500"
    assert list(row) == FIELDS


def test_native_pair_not_prompt_and_score_is_reference_only(tmp_path):
    (row,) = original_rows(write_pair(tmp_path))
    record = adapt_row(
        "STSB",
        row,
        source={"source_id": "STSB", "file_sha256": "fixture"},
        spec=adapter_spec("STSB"),
        index=0,
    )
    assert record.task["input"] == {
        "sentence1": row["sentence1"],
        "sentence2": row["sentence2"],
    }
    assert record.outcome_record() == {"score": "4.500"}
    assert {
        **record.task["input"],
        **record.outcome_record(),
        **record.replay["annotations"],
    } == row
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


@pytest.mark.parametrize("score", ["nan", "inf", "-0.1", "5.01", "not-a-number"])
def test_invalid_continuous_labels_fail_instead_of_coercion(tmp_path, score):
    with pytest.raises(ValueError):
        list(original_rows(write_pair(tmp_path, score=score)))


def test_unknown_split_and_extra_columns_are_not_silently_discarded(tmp_path):
    with pytest.raises(ValueError, match="split and sentence pair"):
        list(original_rows(write_pair(tmp_path, split="validation")))
    with pytest.raises(ValueError, match="width mismatch"):
        list(original_rows(write_pair(tmp_path, extra="\tunmapped")))


def test_wrong_filename_is_not_an_implicit_dataset_conversion(tmp_path):
    with pytest.raises(ValueError, match="filename required"):
        list(original_rows(tmp_path / "made-up.csv"))
