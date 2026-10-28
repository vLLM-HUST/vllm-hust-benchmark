"""Observed dependency manifest fixtures; no browser, renderer, task or model."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/workload-native/structeval-browser-py311-observed.txt"


def module():
    spec = importlib.util.spec_from_file_location(
        "browser_environment", ROOT / "scripts/verify_structeval_browser_environment.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_complete_observed_extension_checked_without_importing_runtime(monkeypatch):
    loaded = module()
    pins = dict(
        line.split("==")
        for line in LOCK.read_text().splitlines()
        if line and not line.startswith("#")
    )
    calls = []

    def version(name):
        calls.append(name)
        return pins[name]

    monkeypatch.setattr(loaded.metadata, "version", version)
    assert loaded.verify_pins(LOCK) == pins
    assert len(calls) == 7
    monkeypatch.setattr(loaded.metadata, "version", lambda _: "mismatch")
    with pytest.raises(ValueError, match="dependency differs"):
        loaded.verify_pins(LOCK)


@pytest.mark.parametrize(
    "content",
    ["playwright==1.64.0", "\n".join(["playwright==1.64.0"] * 7), "not a pin"],
)
def test_partial_duplicate_or_unpinned_extensions_rejected(tmp_path, content):
    lock = tmp_path / "lock.txt"
    lock.write_text(content)
    with pytest.raises(ValueError, match="seven-package"):
        module().verify_pins(lock)


def test_recipe_keeps_browser_installation_in_software_build_only():
    text = (
        ROOT / "configs/workload-native/structeval-browser-cpu.Dockerfile"
    ).read_text()
    assert "playwright install --with-deps chromium" in text
    cache_environment = "PLAYWRIGHT_BROWSERS_PATH=/opt/playwright"  # pragma: allowlist secret - public container cache path, not a credential
    assert cache_environment in text
    assert text.rfind("USER 65534:65534") > text.rfind("RUN ")
    assert "run_structeval_original_json" not in text
