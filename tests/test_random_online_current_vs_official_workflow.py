from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "run-random-online-current-vs-official.yml"


def test_workflow_defaults_use_v0180_official_baseline() -> None:
    workflow_text = WORKFLOW_FILE.read_text(encoding="utf-8")

    assert "ws/random-online-official-v0180-*" in workflow_text
    assert (
        "docs/official-baselines/version-rotations/official-ascend-jun-2026-v0180-random-online-qwen25-14b-910b3.json"
        in workflow_text
    )
    assert "OFFICIAL_VLLM_REF: bcf2be96120005e9aea171927f85055a6a5c0cf6" in workflow_text
    assert "OFFICIAL_VLLM_ASCEND_REF: e18643f8a4d5bd9990727654318ad069ea0b56e2" in workflow_text
    assert "vllm-ascend-official-v0180-v0180" in workflow_text
    assert "submissions/official-ascend-jun-2026-v0.18.0-random-online-qwen25-14b-910b3" in workflow_text