"""Model-free audit fixtures; not native task or NPU evidence."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "labbench_saved_audit", ROOT / "scripts/verify_labbench_original_saved.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize("tamper", [None, "request", "score", "missing", "failed"])
def test_saved_replay_rejects_changed_evidence(tmp_path, tamper):
    instance = SimpleNamespace(id="fixed-id", model_dump=lambda: {"id": "fixed-id"})
    metrics = {"accuracy": 0.0, "n_total": 1}
    result = {
        "instance": instance,
        "input": instance,
        "agent_output": "B",
        "correct": False,
    }
    request = {
        "model": "fixture",
        "max_tokens": 42,
        "messages": [{"role": "user", "content": "original fixture"}],
    }
    attempt = {
        "request": request,
        "response": {"choices": [{"message": {"content": "B"}}]},
    }
    if tamper == "request":
        attempt["request"] = {**request, "max_tokens": 1}
    if tamper == "failed":
        attempt["error_type"] = "TimeoutError"
    if tamper != "missing":
        (tmp_path / "TableQA-attempt-00000.private.json").write_text(
            json.dumps(attempt)
        )
    saved = {
        "metrics_all": metrics,
        "results": {
            "fixed-id": {
                **result,
                "instance": instance.model_dump(),
                "input": instance.model_dump(),
            }
        },
    }
    if tamper == "score":
        saved["metrics_all"] = {"accuracy": 1.0, "n_total": 1}
    (tmp_path / "TableQA-results.private.json").write_text(json.dumps(saved))
    (tmp_path / "TableQA-agent-buffer.private.json").write_text("[]")

    async def close():
        pass

    class Agent:
        def __init__(self, **kwargs):
            self.task_buffer = []
            self.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace()), close=close
            )

        async def run_task(self, _):
            response = await self.client.chat.completions.create(**request)
            return response.choices[0].message.content

    class Evaluator:
        def __init__(self, _):
            self.eval_set = SimpleNamespace(instances=[("fixed-subset", instance)])

        async def score_agent(self, agent_fn, n_threads):
            assert await agent_fn(instance) == "B"
            return {"metrics_all": metrics, "results": {"fixed-id": result}}

    publisher = SimpleNamespace(
        Evaluator=Evaluator, Eval=lambda category: category, OpenAIZeroShotAgent=Agent
    )
    receipt = {
        "per_subset": 1,
        "model": "fixture",
        "declared_output_budget": 42,
        "thinking_override": None,
        "sample_seed": 1,
        "categories": [
            {"category": "TableQA", "selected_ids": ["fixed-id"], "metrics": metrics}
        ],
    }
    if tamper:
        with pytest.raises(ValueError):
            asyncio.run(AUDIT.audit_category(publisher, "TableQA", receipt, tmp_path))
    else:
        report = asyncio.run(
            AUDIT.audit_category(publisher, "TableQA", receipt, tmp_path)
        )
        assert report["saved_tasks_verified"] == 1
        assert report["metrics"]["accuracy"] == 0.0
