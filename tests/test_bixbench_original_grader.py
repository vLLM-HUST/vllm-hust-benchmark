import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bix_grader",
    Path(__file__).resolve().parents[1] / "scripts/run_bixbench_original_grader.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def options(tmp_path, **changes):
    values = dict(
        author_root=tmp_path / "author",
        input_file=tmp_path / "input.csv",
        output=tmp_path / "output",
        model="openai/Qwen/Qwen3-14B",
        api_base="http://127.0.0.1:14012/v1",
    )
    values.update(changes)
    return argparse.Namespace(**values)


def test_plan_does_not_import_author_or_create_output(tmp_path, capsys):
    args = options(tmp_path)
    argv = [
        item
        for key, value in vars(args).items()
        for item in ("--" + key.replace("_", "-"), str(value))
    ]
    assert MODULE.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PLAN_ONLY"
    assert not report["scoring_returned"]
    assert not args.output.exists()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1",
        "http://127.0.0.1/v1",
        "http://user@127.0.0.1:12/v1",
        "http://127.0.0.1:12/v1?x=y",
    ],
)
def test_external_or_ambiguous_endpoint_rejected(tmp_path, endpoint):
    with pytest.raises(ValueError):
        MODULE.validate(options(tmp_path, api_base=endpoint))


def test_no_publisher_or_input_overwrite(tmp_path):
    with pytest.raises(ValueError):
        MODULE.validate(options(tmp_path, output=tmp_path / "author" / "results"))
    with pytest.raises(ValueError):
        MODULE.validate(options(tmp_path, output=tmp_path / "input.csv"))


def test_queue_plan_does_not_create_transport_or_output(tmp_path, capsys):
    args = options(tmp_path, model_queue=tmp_path / "queue")
    argv = [
        item
        for key, value in vars(args).items()
        for item in ("--" + key.replace("_", "-"), str(value))
    ]
    assert MODULE.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_queue_enabled"] and not report["scoring_returned"]
    assert not list(tmp_path.iterdir())


def test_queue_cannot_overlap_inputs_or_output(tmp_path):
    for queue in (
        tmp_path / "author" / "queue",
        tmp_path / "output" / "queue",
        tmp_path,
    ):
        with pytest.raises(ValueError, match="external queue"):
            MODULE.validate(options(tmp_path, model_queue=queue))
