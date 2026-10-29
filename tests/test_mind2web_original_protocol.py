"""Guard reviewed native semantics before an execution adapter is qualified."""

import json
from pathlib import Path


def test_mind2web_protocol_keeps_original_execution_requirements():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs/workload-native/mind2web-original-protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == "protocol_review_only_not_native_validated"
    assert len(protocol["author_revision"]) == 40
    assert protocol["candidate_protocol"]["top_k"] == 50
    assert protocol["candidate_protocol"]["ground_truth_injection"] is False
    assert protocol["verified_author_behavior"]["generation_max_new_tokens"] == 50
    assert protocol["required_splits"] == [
        "test_task",
        "test_website",
        "test_domain",
    ]
    assert (
        protocol["environment_constraints"]["install_into_serving_environment"] is False
    )
    assert len(protocol["remaining"]) == 5
