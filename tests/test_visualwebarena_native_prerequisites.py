"""Invented integrity controls, not original site or Agent execution."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vwa_prerequisites", ROOT / "scripts/check_visualwebarena_native_prerequisites.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(root, *, ids=(0, 1, 2), launcher=None):
    texts = {
        "run.py": launcher
        or (
            "if DATASET == 'visualwebarena':\n"
            "    eval_caption_image_fn = image_utils.get_captioning_fn(\n"
            'choices=["cpu", "cuda"]\n'
        ),
        "environment_docker/README.md": "invented fixture deployment only",
    }
    for name in runner.TASK_FILES:
        texts[name] = json.dumps(
            [
                {
                    "task_id": i,
                    "sites": ["classifieds"],
                    "eval": {"eval_types": ["url_match"]},
                    "image": None,
                    "require_reset": False,
                }
                for i in ids
            ]
        )
    entries = []
    for name, text in texts.items():
        raw = text.encode()
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entries.append(
            dict(
                path=name,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                git_blob=hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            )
        )
    pins = root / "fixture-pins.json"
    pins.write_text(
        json.dumps(
            dict(source_id="VISUALWEBARENA", revision="invented-fixture", files=entries)
        )
    )
    return pins


def test_url_metrics_do_not_hide_original_eager_captioner_requirement(tmp_path):
    result = runner.inspect_original(tmp_path, pins_path=fixture(tmp_path))
    assert result["original_task_definitions"] == 9
    assert [r["task_id"] for r in result["fixed_original_tasks"]] == [0, 1, 2]
    assert all(r["eval_types"] == ["url_match"] for r in result["fixed_original_tasks"])
    assert result["original_launcher_initializes_eval_captioner_even_for_url_match"]
    assert result["original_cli_eval_captioner_devices"] == ["cpu", "cuda"]
    assert not result["npu_captioner_execution_qualified"]
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]


def test_tampered_original_launcher_rejected(tmp_path):
    pins = fixture(tmp_path)
    (tmp_path / "run.py").write_text("modified")
    with pytest.raises(ValueError):
        runner.inspect_original(tmp_path, pins_path=pins)


def test_changed_upstream_captioner_contract_requires_review(tmp_path):
    pins = fixture(tmp_path, launcher="invented changed source")
    with pytest.raises(ValueError, match="captioner/device contract"):
        runner.inspect_original(tmp_path, pins_path=pins)


@pytest.mark.parametrize("ids", [(0, 0, 2), (1, 2, 3)])
def test_no_duplicate_or_reselected_fixed_tasks(tmp_path, ids):
    pins = fixture(tmp_path, ids=ids)
    with pytest.raises(ValueError, match="distinct original|fixed original"):
        runner.inspect_original(tmp_path, pins_path=pins)


@pytest.mark.parametrize(
    "observation", ["image_som", "image", "accessibility_tree_with_captioner"]
)
def test_prepared_captioner_binding_on_both_original_branches(
    tmp_path, monkeypatch, observation
):
    source = """def parser_stub(parser):
    parser.add_argument("--eval_captioning_model_device", choices=["cpu", "cuda"],)

def initialize(args):
    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None
    if DATASET == 'visualwebarena':
        if caption_image_fn and args.eval_captioning_model == args.captioning_model:
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,
            )
    return caption_image_fn, eval_caption_image_fn
"""
    patched = runner.patch_captioner_device(source)
    with pytest.raises(ValueError, match="binding changed"):
        runner.patch_captioner_device(patched)
    local = tmp_path / "blip2"
    local.mkdir()
    monkeypatch.setenv("VWA_BLIP2_ROOT", str(local))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    calls = []
    captioner = object()

    def get_captioning_fn(*args):
        calls.append(args)
        return captioner

    npu = SimpleNamespace(device_count=lambda: 1)
    namespace = {
        "DATASET": "visualwebarena",
        "Path": Path,
        "os": os,
        "torch": SimpleNamespace(device=str, float32="fp32", npu=npu),
        "image_utils": SimpleNamespace(get_captioning_fn=get_captioning_fn),
    }
    exec(compile(patched, "fixture", "exec"), namespace)
    args = SimpleNamespace(
        observation_type=observation,
        captioning_model="Salesforce/blip2-flan-t5-xl",
        eval_captioning_model="Salesforce/blip2-flan-t5-xl",
        eval_captioning_model_device="cpu",
    )
    with pytest.raises(ValueError, match="requires"):
        namespace["initialize"](args)
    args.eval_captioning_model_device = "npu:0"
    npu.device_count = lambda: 2
    with pytest.raises(RuntimeError, match="exactly one"):
        namespace["initialize"](args)
    assert calls == []
    npu.device_count = lambda: 1
    observation_fn, eval_fn = namespace["initialize"](args)
    assert eval_fn is captioner
    assert observation_fn is (None if observation == "image" else captioner)
    assert calls == [("npu:0", "fp32", str(local.resolve()))]


def test_npu_captioner_component_does_not_mark_browser_source_delivered():
    docs = ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    entry = ledger["delivery_inventory"]["VISUALWEBARENA"]
    component = entry["original_caption_helper_npu_component_20261107"]
    receipt = json.loads((ROOT / component["receipt"]).read_text("utf-8"))
    assert not entry["checked"]
    assert not component["source_checkbox_eligible"]
    assert not receipt["source_checkbox_eligible"]
    assert receipt["native_browser_tasks_run"] == receipt["native_scores_produced"] == 0
    assert receipt["control"]["model_calls"] == 1
    assert receipt["control"]["original_max_new_tokens"] == 32
    assert receipt["control"]["prompt_argument"] is None
    assert receipt["original_helper_unmodified"]
    assert receipt["compatibility_profile"]["all_parameter_devices"] == ["npu:0"]
    assert not receipt["compatibility_profile"]["cpu_neural_fallback"]
    assert len(receipt["weight_shards"]) == 2
    assert sum(row["bytes"] for row in receipt["weight_shards"]) == 15769954544


def test_som_template_preparation_requires_audited_input_and_new_output(
    tmp_path, monkeypatch
):
    import prepare_visualwebarena_qwen35_som_template as template

    source = tmp_path / "source.jinja"
    output = tmp_path / "output.jinja"
    raw = ("unchanged before\n" + template.OLD + "\nunchanged after").encode()
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="Unverified"):
        template.prepare(source, output)
    assert not output.exists()
    monkeypatch.setattr(template, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    result = template.prepare(source, output)
    assert source.read_bytes() == raw
    assert output.read_bytes() == raw.replace(
        template.OLD.encode(), template.NEW.encode(), 1
    )
    assert (
        result["output_template_sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert not result["source_checkbox_eligible"]
    with pytest.raises(FileExistsError):
        template.prepare(source, output)


def test_local_som_actor_adaptation_is_explicit_and_preserves_original_actors(
    tmp_path, monkeypatch
):
    actor = (
        "def is_multimodal(lm_config):\n"
        '    return ("gemini" in lm_config.model or "gpt-4" in lm_config.model and '
        '"vision" in lm_config.model)\n'
    )
    tokenizer = """from typing import Any
import tiktoken

class Tokenizer:
    def __init__(self, provider, model_name):
        if provider == "openai":
            self.tokenizer = tiktoken.encoding_for_model(model_name)
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
"""
    original = dict(zip(runner.LOCAL_ACTOR_FILES, (actor, tokenizer)))
    patched = runner.patch_local_actor(original)
    with pytest.raises(ValueError, match="multimodal actor check changed"):
        runner.patch_local_actor({**original, "agent/agent.py": "pass"})
    namespace = {}
    exec(patched["agent/agent.py"], namespace)
    is_multimodal = namespace["is_multimodal"]
    assert is_multimodal(
        SimpleNamespace(provider="openai", mode="chat", model="Qwen/Qwen3.5-9B")
    )
    assert not is_multimodal(
        SimpleNamespace(provider="openai", mode="completion", model="Qwen/Qwen3.5-9B")
    )
    assert is_multimodal(
        SimpleNamespace(provider="openai", mode="chat", model="gpt-4-vision")
    )
    standard = SimpleNamespace(encode=lambda text: [1])
    monkeypatch.setitem(
        sys.modules, "tiktoken", SimpleNamespace(encoding_for_model=lambda _: standard)
    )
    calls = []

    def load(root, **kwargs):
        calls.append((root, kwargs))
        return SimpleNamespace(
            encode=lambda text, *, add_special_tokens: [int(add_special_tokens)]
        )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    monkeypatch.setenv("VWA_QWEN35_TOKENIZER_ROOT", str(tmp_path))
    exec(patched["llms/tokenizers.py"], namespace)
    cls = namespace["Tokenizer"]
    assert cls("openai", "gpt-4-vision")("example") == [1]
    assert calls == []
    local = cls("openai", "Qwen/Qwen3.5-9B")
    assert local.encode("example") == local("example") == [0]
    assert calls == [
        (tmp_path.resolve(), {"local_files_only": True, "trust_remote_code": False})
    ]
