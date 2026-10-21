"""The migration must not import the research project's observation subsystem."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "workload-preparation"
DATA_MODULES = {
    "__init__",
    "adapters",
    "assets",
    "bindings",
    "catalog",
    "consumer",
    "execution",
    "identity",
    "input_views",
    "mapping_contract",
    "pipeline",
    "provider",
    "records",
    "registry",
    "taxonomy",
}


def test_only_data_modules_are_included():
    actual = {path.stem for path in (ROOT / "src/vllm_hust_workload_data").glob("*.py")}
    assert actual == DATA_MODULES


def test_no_observation_dependencies_or_hooks():
    requirements = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert all(
        name not in requirements for name in ("openai", "httpx", "psutil", "torch")
    )
    assert not list((ROOT / "tools").rglob("*.py"))
    assert not (ROOT / "schemas/full_cycle_trace_v1.schema.json").exists()
