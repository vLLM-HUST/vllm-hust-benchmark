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


def test_native_delivery_receipt_keeps_fixed_selection_negative_results_and_scope():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    current = ledger["delivery_inventory"]["LABBENCH"]
    evidence = json.loads((ROOT / current["delivery_receipt"]).read_text("utf-8"))
    assert current["checked"] and current["status"] == "AVAILABLE"
    assert evidence["source_id"] == "LABBENCH" and evidence["source_checkbox_eligible"]
    assert evidence["selected_tasks"] == 93
    categories = evidence["official_categories"]
    assert [item["category"] for item in categories] == list(AUDIT.CATEGORIES)
    assert [item["saved_tasks_verified"] for item in categories] == [
        3,
        3,
        3,
        3,
        45,
        30,
        3,
        3,
    ]
    assert evidence["finish_reasons"] == {"stop": 87, "length": 6}
    assert (
        evidence["parsed_answers"]
        == sum(item["parsed_answers"] for item in categories)
        == 45
    )
    assert (
        evidence["correct_answers"]
        == sum(
            round(item["metrics"]["accuracy"] * item["saved_tasks_verified"])
            for item in categories
        )
        == 14
    )
    assert evidence["original_public_rows_preserved_and_indexed"] == 1967
    assert evidence["fixed_research_readbacks_reused"] == 124
    assert evidence["verified_installed_payloads"] == 46
    assert evidence["audit_new_model_calls"] == evidence["valid_generations_rerun"] == 0
    assert evidence["original_author_cot_prompt_retained"] is True
    assert evidence["thinking_override"] is False
    assert not evidence["quality_acceptance_or_author_model_reproduction"]
    assert not evidence["performance_claim"]
    assert not evidence["raw_data_images_or_model_outputs_published"]


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
