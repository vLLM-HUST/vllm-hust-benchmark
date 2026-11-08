import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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


@pytest.mark.parametrize("patched", [False, True])
def test_plan_does_not_import_author_or_create_output(tmp_path, capsys, patched):
    args = options(tmp_path)
    argv = [
        item
        for key, value in vars(args).items()
        for item in ("--" + key.replace("_", "-"), str(value))
    ]
    assert MODULE.main(argv + (["--patched-grader"] if patched else [])) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PLAN_ONLY"
    assert report["grader_variant"] == (
        "local_patch" if patched else "publisher_original"
    )
    assert not report["scoring_returned"]
    assert not report["scorer_validity_verified"]
    assert not report["source_checkbox_eligible"]
    assert report["known_publisher_scorer_defects"] == [
        "empty_answer_can_receive_partial_match",
        "dedicated_numeric_range_template_can_be_bypassed",
    ]
    assert not args.output.exists()


def test_wrong_retained_input_digest_stops_before_scoring(tmp_path, monkeypatch):
    args = options(tmp_path, input_sha256="0" * 64)
    args.input_file.write_text("question,target,predicted\nq,t,p\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "prepare_jsonschemabench_author_source",
        SimpleNamespace(verify_source=lambda *_: None),
    )
    with pytest.raises(ValueError, match="retained SHA-256"):
        asyncio.run(MODULE.execute(args, {}))
    assert not args.output.exists()


def test_patch_loader_only_replaces_grader(tmp_path):
    patch = tmp_path / "graders.py"
    patch.write_text("marker = 'local patch'\n", encoding="utf-8")
    loader = MODULE.PatchedGraderLoader(patch)
    assert loader.find_spec("bixbench.other") is None
    spec = loader.find_spec("bixbench.graders")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert loader.loaded and module.marker == "local patch"


def test_patched_scoring_routes_publisher_package_import(tmp_path, monkeypatch):
    args = options(tmp_path, patched_grader=True)
    pkg = args.author_root / "bixbench"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .graders import marker\n")
    (pkg / "graders.py").write_text("marker = 'original'\n")
    (args.author_root / "grade_outputs.py").write_text(
        "from bixbench import marker\n"
        "class AnswerMode:\n    openanswer = 'openanswer'\n"
        "async def grade_answers(*args, **kwargs):\n"
        "    return {'marker': marker}\n"
    )
    args.input_file.write_text("question,target,predicted\nq,a,\n")

    def prepare(author, overlay):
        replacement = overlay / "bixbench/graders.py"
        replacement.parent.mkdir(parents=True)
        replacement.write_text("marker = 'patched'\n")
        return {"official_unmodified_score": False}

    monkeypatch.setitem(
        sys.modules,
        "prepare_jsonschemabench_author_source",
        SimpleNamespace(verify_source=lambda *args: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "prepare_bixbench_patched_grader",
        SimpleNamespace(SOURCE="bixbench/graders.py", prepare=prepare),
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(MODULE.os, "environ", MODULE.os.environ.copy())
    try:
        result = asyncio.run(MODULE.execute(args, {}))
    finally:
        sys.modules.pop("bixbench", None)
    assert result["status"] == "PATCHED_OPENANSWER_SCORING_RETURNED"
    assert json.loads((args.output / "metrics.private.json").read_text()) == {
        "marker": "patched"
    }
    assert (pkg / "graders.py").read_text() == "marker = 'original'\n"


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
