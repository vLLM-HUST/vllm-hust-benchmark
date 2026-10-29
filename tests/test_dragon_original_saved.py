"""Tiny saved-protocol fixtures; no dataset, tokenizer/model import or NPU use."""

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "dragon_saved", SCRIPTS / "verify_dragon_original_saved.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_original_chunks_match_in_original_id_order_without_dedup():
    choices = {1: ["first", "alternative"], 2: ["second\nparagraph", "other"]}
    assert runner.bind_context("first\n\nsecond\nparagraph", (1, 2), choices) == (
        "first",
        "second\nparagraph",
    )
    assert runner.bind_context("first\n\nfirst", (1, 1), choices) == ("first", "first")
    with pytest.raises(ValueError, match="exact original"):
        runner.bind_context("second\nparagraph\n\nfirst", (1, 2), choices)
    with pytest.raises(ValueError, match="exact original"):
        runner.bind_context("first\n\nreference answer inserted", (1, 2), choices)


def test_chunk_prefix_ambiguity_is_resolved_without_truncating():
    choices = {1: ["a", "a\n\nb"], 2: ["c"]}
    assert runner.bind_context("a\n\nb\n\nc", (1, 2), choices) == ("a\n\nb", "c")


def test_saved_vectors_bound_to_complete_original_batch_and_dimension():
    record = {
        "texts": ["original\ntext"],
        "encode_kwargs": {},
        "embeddings": [[0.1, 0.2]],
    }
    runner.verify_vectors(record, ["original\ntext"], 2)
    with pytest.raises(ValueError, match="input order"):
        runner.verify_vectors(record, ["rewritten text"], 2)
    with pytest.raises(ValueError, match="dimensions"):
        runner.verify_vectors(record, ["original\ntext"], 3)
    record["embeddings"] = [[float("nan"), 0.1]]
    with pytest.raises(ValueError, match="finite"):
        runner.verify_vectors(record, ["original\ntext"], 2)


def test_template_is_extracted_from_author_ast_without_executing_it(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "calculate_baseline_metrics.py").write_text(
        "llm_system_prompt = 'original system'\n"
    )
    source = tmp_path / "lib/src/rag_bench"
    source.mkdir(parents=True)
    (source / "baseline.py").write_text(
        "def init_generation():\n    messages = [{'role':'system','content':llm_prompt}, {'role':'user','content':'Context: {context}; Question: {input}'}]\n    raise RuntimeError('must never execute')\n"
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {"role": "system", "content": "original system"}
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return messages[1]["content"]

    assert (
        runner.original_template(tmp_path, Tokenizer())
        == "Context: {context}; Question: {input}"
    )
