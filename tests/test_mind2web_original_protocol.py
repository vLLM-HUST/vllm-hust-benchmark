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
    assert protocol["status"] == "fixed_original_native_sample_and_consumer_validated"
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
    assert protocol["remaining"] == []
    receipt = json.loads((root / protocol["delivery_receipt"]).read_text("utf-8"))
    assert receipt["source_checkbox_eligible"]
    assert receipt["native_model_requests"] == 846
    assert receipt["saved_response_public_runner_replay"]["new_model_calls"] == 0
