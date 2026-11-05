import importlib.util
import asyncio
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "swesmith_original_agent",
    Path(__file__).resolve().parents[1] / "scripts/run_swesmith_original_agent.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def bindings():
    return {
        identity: {
            "host": "http://172.18.0.2",
            "port": 8000,
            "auth_token": "fixture-authentication-not-a-secret",
            "timeout": 600,
        }
        for identity in MODULE.COMMITS
    }


def test_explicit_retry_preserves_frozen_order_and_never_changes_task_pool():
    ids = list(MODULE.COMMITS)
    assert MODULE.select_task_ids(None) == ids
    assert MODULE.select_task_ids([ids[2], ids[0]]) == [ids[0], ids[2]]
    for requested in [[], [ids[0], ids[0]], ["replacement-easier-task"]]:
        with pytest.raises(ValueError):
            MODULE.select_task_ids(requested)


def test_private_native_runtime_bindings_preserve_exact_task_set():
    MODULE.validate_bindings(bindings())
    with pytest.raises(ValueError, match="three frozen"):
        MODULE.validate_bindings({})


@pytest.mark.parametrize(
    "host",
    [
        "http://1.1.1.1",
        "https://172.18.0.2",
        "http://0.0.0.0",
        "http://172.18.0.2/path",
        "http://name:password@172.18.0.2",  # pragma: allowlist secret - inert rejected-URL fixture, not real credentials
    ],
)
def test_runtime_binding_rejects_public_or_credential_urls(host):
    rows = bindings()
    rows[next(iter(rows))]["host"] = host
    with pytest.raises(ValueError):
        MODULE.validate_bindings(rows)


def test_remote_health_adapter_preserves_result_and_actual_deadline():
    deployment = object()
    marker = object()

    async def original(value):
        assert value is deployment
        return marker

    assert (
        asyncio.run(MODULE.remote_health_with_timeout(original, deployment)) is marker
    )
    assert (
        asyncio.run(MODULE.remote_health_with_timeout(original, deployment, 10))
        is marker
    )

    async def slow(value):
        await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        asyncio.run(MODULE.remote_health_with_timeout(slow, deployment, 0.001))
    with pytest.raises(ValueError, match="positive"):
        asyncio.run(MODULE.remote_health_with_timeout(original, deployment, 0))


def test_remote_health_adapter_does_not_turn_runtime_failure_into_success():
    async def failed(value):
        raise RuntimeError("native health failed")

    with pytest.raises(RuntimeError, match="native health failed"):
        asyncio.run(MODULE.remote_health_with_timeout(failed, object(), 10))
