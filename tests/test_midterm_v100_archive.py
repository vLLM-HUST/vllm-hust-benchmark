"""Historical regeneration must never overwrite the current delivery report."""

import runpy
from pathlib import Path

import pytest


def generator(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    prepare = runpy.run_path(str(scripts / "build_midterm_v100.py"))["prepare"]
    namespace = prepare.__globals__
    monkeypatch.setitem(namespace, "ROOT", tmp_path)
    monkeypatch.setitem(namespace, "render_report", lambda *args: "V1.00 archive\n")
    current = tmp_path / namespace["previous"].REPORT
    current.parent.mkdir(parents=True)
    current.write_text("V1.03 current delivery\n")
    for source in (namespace["SOURCE"], namespace["previous"].SOURCE):
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return prepare, namespace, current


def test_prepare_retains_current_delivery(monkeypatch, tmp_path):
    prepare, namespace, current = generator(monkeypatch, tmp_path)
    (tmp_path / namespace["OLD_REPORT"]).write_text("V0.99 frozen source")
    prepare()
    assert current.read_text() == "V1.03 current delivery\n"
    assert (tmp_path / namespace["REPORT"]).read_text() == "V1.00 archive\n"


def test_missing_frozen_source_does_not_reuse_current_report(monkeypatch, tmp_path):
    prepare, namespace, current = generator(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="Frozen V0.99"):
        prepare()
    assert current.read_text() == "V1.03 current delivery\n"
    assert not (tmp_path / namespace["REPORT"]).exists()
