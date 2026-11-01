"""Isolation software fixtures, not original tasks or model-generated programs."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "discovery_tool_boundary",
    Path(__file__).resolve().parents[1] / "scripts/discoverybench_isolated_python.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def state():
    return dict(
        container=True,
        uid=65534,
        status="CapEff:\t00000000\nNoNewPrivs:\t1\n",
        interfaces=["lo"],
        readonly=True,
    )


def test_restricted_fixture_accepted_without_importing_or_running_code():
    assert entry.validate_isolation(**state()) is None


@pytest.mark.parametrize(
    "change",
    [
        {"container": False},
        {"uid": 0},
        {"readonly": False},
        {"interfaces": ["lo", "eth0"]},
        {"status": "CapEff:\t00000001\nNoNewPrivs:\t1\n"},
        {"status": "CapEff:\t00000000\nNoNewPrivs:\t0\n"},
        {"status": ""},
    ],
)
def test_weaker_boundary_rejected(change):
    values = state()
    values.update(change)
    with pytest.raises(RuntimeError):
        entry.validate_isolation(**values)
