import base64
import importlib.util
from pathlib import Path


def test_interleaved_content_preserves_loader_order():
    script = Path(__file__).resolve().parents[1] / "scripts/verify_longvideobench_original_inputs.py"
    spec = importlib.util.spec_from_file_location("longvideobench_inputs", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class Frame:
        def save(self, buffer, *, format):
            assert format == "PNG"
            buffer.write(b"frame-payload")

    parts = list(runner.interleaved_content(["subtitle", Frame(), "question and options"]))
    assert [part["type"] for part in parts] == ["text", "image_url", "text"]
    assert [parts[0]["text"], parts[2]["text"]] == ["subtitle", "question and options"]
    assert base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]) == b"frame-payload"
