import inspect
from types import SimpleNamespace

import pytest

from scripts.run_vllm_cli_compat import (
    install_offline_graph_guard,
    offline_graph_proof,
    require_offline_graph,
)


def _fake_llm(*, enforce_eager: bool, mode: str, cudagraph_mode: str) -> object:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name=mode),
            cudagraph_mode=SimpleNamespace(name=cudagraph_mode),
        ),
    )
    return SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config))


def test_offline_graph_proof_accepts_effective_piecewise_graph() -> None:
    proof = offline_graph_proof(
        _fake_llm(
            enforce_eager=False,
            mode="VLLM_COMPILE",
            cudagraph_mode="PIECEWISE",
        )
    )

    require_offline_graph(proof)
    assert proof["graph_mode_verified"] is True


@pytest.mark.parametrize(
    ("enforce_eager", "mode", "cudagraph_mode"),
    [
        (True, "NONE", "NONE"),
        (False, "NONE", "PIECEWISE"),
        (False, "VLLM_COMPILE", "NONE"),
    ],
)
def test_offline_graph_proof_rejects_eager_or_disabled_graph(
    enforce_eager: bool,
    mode: str,
    cudagraph_mode: str,
) -> None:
    proof = offline_graph_proof(
        _fake_llm(
            enforce_eager=enforce_eager,
            mode=mode,
            cudagraph_mode=cudagraph_mode,
        )
    )

    with pytest.raises(RuntimeError, match="eager/non-graph"):
        require_offline_graph(proof)


def test_offline_graph_guard_supports_pre_from_engine_args_llm_api() -> None:
    source = inspect.getsource(install_offline_graph_guard)

    assert 'hasattr(LLM, "from_engine_args")' in source
    assert "original_init = LLM.__init__" in source
    assert "record_proof(self)" in source
