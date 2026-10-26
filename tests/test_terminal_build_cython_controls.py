"""Keep failed environment controls separate from model-task results."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/terminal-build-cython-controls-20261027"
SPEC = importlib.util.spec_from_file_location(
    "cython_controls_verify", DATA / "verify.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def inputs():
    return json.loads((DATA / "summary.json").read_text()), json.loads(
        (DATA / "independent-review.json").read_text()
    )


def test_original_control_results_and_hashes():
    VERIFY.main()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(control_qualification_passed=True),
        lambda s: s.update(new_model_task_attempts=1),
        lambda s: s.update(original_amd64_image_equivalence_claimed=True),
        lambda s: s["controls"]["oracle"].update(reward=1),
        lambda s: s["controls"]["oracle"]["tests"].pop(),
        lambda s: s["budget"].update(verifier_seconds=1800),
        lambda s: s["controls"]["oracle"].update(all_owned_containers_released=False),
    ],
)
def test_misleading_control_claims_rejected(mutation):
    summary, review = inputs()
    changed = copy.deepcopy(summary)
    mutation(changed)
    with pytest.raises(AssertionError):
        VERIFY.validate(changed, review)
