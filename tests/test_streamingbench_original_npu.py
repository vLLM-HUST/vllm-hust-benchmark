"""Software-only original binding fixtures, no actual model/video validation."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "streaming_npu", ROOT / "scripts/run_streamingbench_original_npu.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

RAW = """
model = Box.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto", attn_implementation="flash_attention_2", torch_dtype="auto")
processor = Box.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
inputs = inputs.to("cuda")
answer = model.generate(max_new_tokens=128)
"""


@pytest.mark.parametrize("attention", ["sdpa", "eager"])
def test_only_checkpoint_device_and_attention_are_rebound(attention):
    loads = []

    class Box:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loads.append((path, kwargs))
            return SimpleNamespace(generate=lambda **options: options)

    namespace = {"Box": Box, "inputs": SimpleNamespace(to=lambda device: device)}
    exec(runner.npu_program(RAW, Path("/owned/model"), "npu:2", attention), namespace)
    assert len(loads) == 2
    assert all(
        path == str(Path("/owned/model").resolve()) and args["local_files_only"]
        for path, args in loads
    )
    assert "device_map" not in loads[0][1]
    assert loads[0][1]["attn_implementation"] == attention
    assert loads[0][1]["torch_dtype"] == "auto"
    assert namespace["inputs"] == "npu:2"
    assert namespace["answer"] == {"max_new_tokens": 128}


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "npu", "npu:-1"])
def test_other_devices_are_refused(device):
    with pytest.raises(ValueError, match="explicit NPU"):
        runner.npu_program(RAW, Path("/owned/model"), device)


def test_unexpected_original_structure_is_not_silently_patched():
    with pytest.raises(ValueError):
        runner.npu_program(
            RAW.replace('"cuda"', '"cpu"'), Path("/owned/model"), "npu:0"
        )
    with pytest.raises(ValueError):
        runner.npu_program(
            RAW.replace('"flash_attention_2"', '"other"'), Path("/owned/model"), "npu:0"
        )


def test_missing_media_plan_keeps_three_groups_sharing_one_video(tmp_path, monkeypatch):
    author = tmp_path / "author"
    (author / "src/model").mkdir(parents=True)
    (author / "src/model/Qwen2VL.py").write_text(RAW)
    data = tmp_path / "data"
    (data / "src/data").mkdir(parents=True)
    rows = [
        {"video_path": "./videos/shared.mp4", "questions": [{"question": str(index)}]}
        for index in range(3)
    ]
    (data / "src/data/questions_real.json").write_text(json.dumps(rows))
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(runner, "verify_source", lambda *args: author)
    monkeypatch.setattr(runner, "verify_file", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "source_spec",
        lambda _: {
            "files": [
                {"path": "src/data/questions_real.json", "sha256": "fixture-only"}
            ]
        },
    )
    monkeypatch.setattr(
        runner,
        "verify_model",
        lambda *args: pytest.fail("plan must not load/check weights"),
    )
    args = SimpleNamespace(
        author_root=author,
        data_root=data,
        media_root=media,
        task="real",
        model_root=tmp_path / "model",
        model_revision=json.loads(runner.MODEL_PINS.read_text())["revision"],
        output=tmp_path / "output",
        execute=False,
        full=False,
        device="npu:0",
    )
    report = runner.run(args)
    assert report["selected_complete_outer_groups"] == 3
    assert report["original_question_occurrences"] == 3
    assert report["missing_original_media"] == ["shared.mp4"]
    assert report["native_model_calls"] == 0 and not report["model_artifacts_verified"]
    assert report["attention_binding"] == "sdpa replaces CUDA FlashAttention2"
    assert not args.output.exists()


def test_complete_nested_groups_and_source_paths():
    rows = [
        [{"video_path": "./videos/a.mp4", "questions": [{"question": "original"}]}],
        {"video_path": "./videos/b.mp4", "questions": []},
    ]
    assert runner.media_names(rows) == ["a.mp4", "b.mp4"]
    for path in ("../videos/a.mp4", "https://example.com/a.mp4", "videos/a.txt"):
        with pytest.raises(ValueError):
            runner.media_names([{"video_path": path, "questions": []}])
    with pytest.raises(ValueError, match="silently skip"):
        runner.media_names(
            [
                {
                    "video_path": "videos/a.mp4",
                    "questions": [{runner.RESULT_KEY: "cached"}],
                }
            ]
        )


def prefix_fixture(tmp_path):
    author = tmp_path / "author"
    (author / "src/utils").mkdir(parents=True)
    utility = author / "src/utils/video_execution.py"
    utility.write_bytes(b"original utility fixture, not executable author code")
    workspace = tmp_path / "prepared"
    directory = workspace / "videos/tmp_60"
    directory.mkdir(parents=True)
    rows, clips = [], []
    for timestamp in [31, 206, 403]:
        name = f"sample_348_real_0_{timestamp}.mp4"
        raw = f"encoded-video fixture {timestamp}".encode()
        (directory / name).write_bytes(raw)
        clips.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "start_seconds": 0,
                "end_seconds": timestamp,
            }
        )
        rows.append(
            {
                "video_path": "./videos/sample_348_real.mp4",
                "questions": [
                    {"time_stamp": f"00:{timestamp // 60:02d}:{timestamp % 60:02d}"}
                ],
            }
        )
    registered = {"sample_348_real.mp4": {"sha256": "0" * 64}}
    saved = {
        "schema": "streamingbench-original-prefix-cache/v1",
        "source_id": "STREAMINGBENCH",
        "task": "real",
        "selection": "first3_original_complete_outer_groups",
        "context_time": -1,
        "source_question_sha256": "question-fixture",
        "split_video_code_sha256": hashlib.sha256(utility.read_bytes()).hexdigest(),
        "source_media_sha256": {"sample_348_real.mp4": "0" * 64},
        "directory": "videos/tmp_60",
        "clips": clips,
    }
    path = workspace / "prefix-cache.json"
    path.write_text(json.dumps(saved))
    return author, directory, rows, registered, saved, path


def test_original_prepared_prefix_integrity_and_time_selection(tmp_path):
    author, directory, rows, registered, saved, path = prefix_fixture(tmp_path)
    assert runner.verify_prefix_cache(
        path, rows, "question-fixture", registered, author
    ) == (directory, 3)
    (directory / saved["clips"][1]["path"]).write_bytes(b"different")
    with pytest.raises(ValueError, match="size/type differs"):
        runner.verify_prefix_cache(path, rows, "question-fixture", registered, author)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_time", 30),
        ("source_question_sha256", "changed"),
        ("directory", "../elsewhere"),
    ],
)
def test_changed_prefix_protocol_is_rejected(tmp_path, field, value):
    author, _, rows, registered, saved, path = prefix_fixture(tmp_path)
    saved[field] = value
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="protocol identity differs"):
        runner.verify_prefix_cache(path, rows, "question-fixture", registered, author)


def test_prefix_clip_cannot_escape_original_filename_or_time(tmp_path):
    author, _, rows, registered, saved, path = prefix_fixture(tmp_path)
    saved["clips"][0]["path"] = "../outside.mp4"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="name/time selection differs"):
        runner.verify_prefix_cache(path, rows, "question-fixture", registered, author)


def test_unsupported_attention_is_not_automatically_substituted():
    with pytest.raises(ValueError, match="supported NPU attention"):
        runner.npu_program(RAW, Path("/owned/model"), "npu:0", "automatic")
