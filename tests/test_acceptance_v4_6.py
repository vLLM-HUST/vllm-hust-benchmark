import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
DECLARATION = REPO_ROOT / "src/vllm_hust_benchmark/data/acceptance_v4_6.json"


def test_v46_acceptance_pdf_is_repository_controlled_and_pinned() -> None:
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    source = REPO_ROOT / declaration["source_document"]

    assert source == REPO_ROOT / "docs/assets/vLLM-HUST标准交付测试方案_V4.6.pdf"
    assert source.is_file()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == declaration["source_sha256"]
    assert declaration["source_custodian"]
    assert (
        declaration["source_verification"]["expected_sha256_field"] == "source_sha256"
    )
