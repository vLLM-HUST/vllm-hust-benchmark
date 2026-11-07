"""Signature fixtures only; no desktop, model or native task execution."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from osworld_qwen3vl_reset_compat import reset_compatible_agent


def test_delegates_original_reset_and_preserves_other_methods():
    class Original:
        def reset(self, _logger=None):
            self.logger = _logger
            self.observations = []
            return "original-reset-return"

        def predict(self):
            raise AssertionError("fixture must not call a model")

    adapted = reset_compatible_agent(Original)
    assert adapted.predict is Original.predict
    agent = adapted()
    logger = object()
    assert agent.reset(logger, vm_ip="127.0.0.1") == "original-reset-return"
    assert agent.logger is logger and agent.observations == []
    agent.reset(vm_ip="127.0.0.1")
    assert agent.logger is None


def test_unknown_upstream_signature_is_not_silently_adapted():
    class Changed:
        def reset(self, another_argument):
            pass

    with pytest.raises(ValueError, match="pinned Qwen3VL"):
        reset_compatible_agent(Changed)


def test_original_errors_propagate():
    class Original:
        def reset(self, _logger=None):
            raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        reset_compatible_agent(Original)().reset(vm_ip="127.0.0.1")


def test_changed_logger_default_is_rejected():
    class Changed:
        def reset(self, _logger):
            pass

    with pytest.raises(ValueError, match="pinned Qwen3VL"):
        reset_compatible_agent(Changed)
