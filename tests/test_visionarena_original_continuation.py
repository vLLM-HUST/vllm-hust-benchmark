import base64
import importlib.util
from pathlib import Path


def test_original_image_and_history_are_preserved_and_target_withheld():
    script = Path(__file__).resolve().parents[1] / "scripts/run_visionarena_original_continuation.py"
    spec = importlib.util.spec_from_file_location("visionarena_continuation", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    messages = [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "historical answer"},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "withheld target"},
    ]
    raw = b"original-image"
    prefix = runner.continuation_prefix(messages, raw)
    assert prefix[1:] == messages[1:3]
    assert prefix[0]["content"][1]["text"] == messages[0]["content"]
    assert base64.b64decode(prefix[0]["content"][0]["image_url"]["url"].split(",")[1]) == raw
    assert messages[0]["content"] == "first prompt"
