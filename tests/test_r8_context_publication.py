import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/r8-context-tenant-20261119"
spec = importlib.util.spec_from_file_location(
    "r8_context_publication", REPORT / "reproduce_publication.py"
)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def source():
    value = json.loads((REPORT / "summary-public-v1.json").read_text())
    pair = dict(
        schema="vllm-hust-paired-context-tenant-supplement/v1",
        execution_complete=True,
        contract=value["protocol"],
        comparison=value["comparisons"],
        arm_seals=value["arm_seals"],
    )
    return value, pair


def test_publication_reconstruction_keeps_all_windows_and_tenant_distributions():
    value, pair = source()
    assert (
        publication.project(
            pair,
            chart_sha256=value["chart"]["sha256"],
            chart_bytes=value["chart"]["bytes"],
        )
        == value
    )


@pytest.mark.parametrize(
    "fault", ["budget", "missing_phase", "tenant_loss", "output_total"]
)
def test_corrupted_source_cannot_be_relabelled_as_a_completed_supplement(fault):
    value, pair = source()
    pair = copy.deepcopy(pair)
    if fault == "budget":
        pair["contract"]["output_tokens"] = 128
    elif fault == "missing_phase":
        pair["comparison"].pop()
    elif fault == "tenant_loss":
        pair["comparison"][-1]["B1"]["per_tenant"].pop("tenant-3")
    elif fault == "output_total":
        pair["comparison"][0]["B0"]["metrics"]["output_tokens"] += 1
    with pytest.raises(ValueError):
        publication.project(
            pair,
            chart_sha256=value["chart"]["sha256"],
            chart_bytes=value["chart"]["bytes"],
        )
