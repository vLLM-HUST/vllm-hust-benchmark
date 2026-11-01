"""Synthetic CPU AST-binding fixtures, not original task/model evidence."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "sab_baseline",
    Path(__file__).resolve().parents[1]
    / "scripts/run_scienceagentbench_original_baseline.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


@pytest.fixture
def binding(tmp_path, monkeypatch):
    source = """
SYSTEM_PROMPT="original system"
SELF_DEBUG_PROMPT="debug"
FORMAT_PROMPT="format"
REQUEST_PROMPT="request"
DATA_INFO_PROMPT="data"
class ScienceAgent:
 def __init__(self): raise RuntimeError("must not initialize remote vendor")
 def install(self): raise RuntimeError("must not install packages")
 def step(self): raise RuntimeError("must not run generated code")
 def get_sys_msg(self, task):
  return trim_messages([{'role':'user','content':task['task_inst']}], self.llm_engine.llm_engine_name, max_tokens=self.context_cutoff-2000)[0]['content']
 def write_program(self,text,path):
  Path(path).write_text(text)
 def solve_task(self,task,out_fname):
  self.sys_msg=self.get_sys_msg(task)
  messages=[{'role':'user','content':self.sys_msg}]
  answer,prompt_tokens,completion_tokens=self.llm_engine.respond(messages,temperature=0.2,top_p=0.95)
  cost=self.llm_cost['input_cost_per_token']*prompt_tokens+self.llm_cost['output_cost_per_token']*completion_tokens
  self.write_program(answer,out_fname)
  return {'history':messages+[{'role':'assistant','content':answer}],'cost':cost}
"""
    infer = "def format_task_dict(row,args):\n return {'task_inst':row['task_inst']}\n"
    hashes = {}
    for name, text in [("agent.py", source), ("run_infer.py", infer)]:
        raw = text.encode()
        (tmp_path / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(entry, "SOURCE_HASHES", hashes)
    calls = []

    def respond(messages, temperature, top_p):
        calls.append((messages, temperature, top_p))
        return "unchanged answer", 10, 20

    engine = SimpleNamespace(llm_engine_name="actual/model", respond=respond)
    return tmp_path, engine, calls


def test_original_baseline_keeps_messages_sampling_and_marks_unpriced_cost(binding):
    root, engine, calls = binding
    generate = entry.original_baseline(root, engine, lambda messages: 100)
    out = root / "prediction.py"
    result = generate({"task_inst": "unchanged original task"}, out)
    assert calls == [
        ([{"role": "user", "content": "unchanged original task"}], 0.2, 0.95)
    ]
    assert result["cost"] is None
    assert out.read_text() == "unchanged answer"


def test_over_budget_fails_before_model_without_truncation(binding):
    root, engine, calls = binding
    generate = entry.original_baseline(root, engine, lambda messages: 26001)
    with pytest.raises(ValueError, match="not truncated"):
        generate({"task_inst": "unchanged original task"}, root / "prediction.py")
    assert calls == []


def test_existing_program_is_not_overwritten_or_regenerated(binding):
    root, engine, calls = binding
    out = root / "prediction.py"
    out.write_text("saved valid answer")
    generate = entry.original_baseline(root, engine, lambda messages: 100)
    with pytest.raises(FileExistsError):
        generate({"task_inst": "task"}, out)
    assert calls == []
    assert out.read_text() == "saved valid answer"


def test_unknown_source_is_rejected_before_compilation(binding):
    root, engine, calls = binding
    (root / "agent.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.original_baseline(root, engine, lambda messages: 100)
    assert calls == []
