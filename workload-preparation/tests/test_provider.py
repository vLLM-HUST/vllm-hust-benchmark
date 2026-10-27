import pytest

from vllm_hust_workload_data import provider
from vllm_hust_workload_data.taxonomy import research_classes


def test_provider_identity_and_resources():
    identity = provider.provider_identity()
    assert identity["version"] == "0.1.0"
    assert identity["contract"] == "benchmark-workload-provider/v1"
    assert identity["provenance"]["raw_datasets_included"] is False


def test_every_class_has_an_explicit_plan():
    classes = research_classes()
    assert len(classes) == 38
    for row in classes:
        plan = provider.class_plan(row["family_id"])
        assert plan["compatibility_family"] == row["parent_family_id"]
        assert plan["selection_scope"] == "CANDIDATE_SOURCES_NOT_RECORD_MEMBERSHIP"
    with pytest.raises(ValueError):
        provider.class_plan("not-a-class")


def test_catalog_registration_does_not_claim_runtime_or_local_data():
    sources = provider.list_sources()
    assert len(sources) == 103  # frozen100 plus three already-planned sources
    assert sum(row["source_id"].startswith("PAIO-") for row in sources) == 8
    assert all(
        row["runtime_validation"] == "NOT_ESTABLISHED_BY_CATALOG" for row in sources
    )
    assert all(
        row["local_availability"] == "NOT_CHECKED_ON_THIS_MACHINE" for row in sources
    )
