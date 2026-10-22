import importlib.util
from pathlib import Path


def test_c16_raw_request_and_weight_identity():
    path = (
        Path(__file__).resolve().parents[1]
        / "reports/c16-baseline018-reuse-20261022/audit.py"
    )
    spec = importlib.util.spec_from_file_location("c16_reuse_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit()
    assert result["b0_requests"] == result["matched_b0_requests"] == 408
    assert result["b1_requests"] == 1296
    assert result["first_turn_hash_differences"] == 0
    assert result["field_differences"] == {
        "trajectory_id": 0,
        "prompt_sha256": 370,
        "prompt_tokens": 0,
        "expected_output_tokens": 0,
    }
    assert result["identical_upstream_weight_shards"] == 14
    assert result["window_tokens"] == [222539, 797365]
