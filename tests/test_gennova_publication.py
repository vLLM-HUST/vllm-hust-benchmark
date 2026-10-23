import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_publication_is_derived_from_sealed_run():
    spec = importlib.util.spec_from_file_location(
        "gennova", ROOT / "scripts/build_gennova_publication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads(module.TARGET.read_text())
    assert payload == module.build()
    assert payload["kind"] == "independent-engine"
    assert payload["b1_replacement_approved"] is False
    assert payload["full_dataset_quality_score"] is None
    assert payload["metrics"]["completed_requests"] == 200
