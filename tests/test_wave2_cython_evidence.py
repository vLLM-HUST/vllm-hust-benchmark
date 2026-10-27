"""Keep original scoring, recovered scoring and unscored attempts distinct."""

import copy
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(folder):
    path = ROOT / "reports" / folder / "verify.py"
    spec = importlib.util.spec_from_file_location(folder.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wave2_original_and_recovery_denominator():
    value = load("swe-native018-wave2-first-three-20261027").verify()
    assert value == {
        "status": "passed",
        "attempts": 21,
        "original_positives": 3,
        "disclosed_recovery_positives": 4,
        "unattempted": 710,
    }


def test_cython_context_failure_stays_unscored():
    value = load("terminal-cython-context-20261027").verify()
    assert value["new_task_reward"] is None
    assert value["reconstructed_artifact_reward"] == 0
    assert value["model_attempts"] == 3 and value["unattempted"] == 86


@pytest.mark.parametrize(
    "key,value",
    [
        ("reward", 0),
        ("official_scored", True),
        ("HTTP_success", 30),
        ("successful_Agent_steps", 30),
        ("Terminal_attempts", 4),
    ],
)
def test_reject_changed_model_claim(monkeypatch, key, value):
    module = load("terminal-cython-context-20261027")
    original = module.read
    changed = copy.deepcopy(original("model-summary.json"))
    changed[key] = value
    monkeypatch.setattr(
        module,
        "read",
        lambda name: changed if name == "model-summary.json" else original(name),
    )
    with pytest.raises(AssertionError):
        module.verify()


def test_readmes_are_generated_from_canonical_sources():
    path = ROOT / "scripts/build_wave2_cython_docs.py"
    spec = importlib.util.spec_from_file_location("wave2_docs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def normalize(text):
        return re.sub(r"-{3,}", "---", re.sub(r"\s+", "", text))

    assert normalize((module.SWE / "README.md").read_text()) == normalize(
        module.build_swe()
    )
    assert normalize((module.TERMINAL / "README.md").read_text()) == normalize(
        module.build_terminal()
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("reward", 1),
        ("original_reward", 0),
        ("additional_model_attempts", 1),
        ("original_native_or_install_snapshot", True),
        ("terminal_model_attempts", 4),
    ],
)
def test_reject_recovery_identity_conflation(monkeypatch, key, value):
    module = load("terminal-cython-context-20261027")
    original = module.read
    changed = copy.deepcopy(original("recovery-summary.json"))
    changed[key] = value
    monkeypatch.setattr(
        module,
        "read",
        lambda name: changed if name == "recovery-summary.json" else original(name),
    )
    with pytest.raises(AssertionError):
        module.verify()
