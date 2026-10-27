"""Noisy auxiliary labels must not silently become human test references."""

import hashlib
from copy import deepcopy

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.registry import source_spec


def test_configuration_and_label_kind_separate_from_original_inputs(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    relative = "unlabeled_final/train-00000-of-00001.parquet"
    path = tmp_path / relative
    path.parent.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": 7,
                    "sentence1": "original A",
                    "sentence2": "original B",
                    "label": 1,
                }
            ]
        ),
        path,
    )
    manifest = deepcopy(source_spec("PAWS"))
    manifest["revision"] = "fixture"
    manifest["files"] = [
        {
            "path": relative,
            "role": "data",
            "rows": 1,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "configuration": "unlabeled_final",
            "reference_kind": "noisy_auxiliary",
        }
    ]
    (record,) = iter_workload("PAWS", data_root=tmp_path, manifest=manifest)
    assert record.source["dataset_configuration"] == "unlabeled_final"
    assert record.source["label_reference_kind"] == "noisy_auxiliary"
    assert record.task["input"] == {
        "sentence1": "original A",
        "sentence2": "original B",
    }
    assert record.outcome_record() == {"label": 1}
    manifest["files"][0]["reference_kind"] = "human"
    with pytest.raises(ValueError):
        list(iter_workload("PAWS", data_root=tmp_path, manifest=manifest))
