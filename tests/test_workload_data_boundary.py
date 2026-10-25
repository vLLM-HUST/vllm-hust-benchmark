"""The migration must not import the research project's observation subsystem."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "workload-preparation"
DATA_MODULES = {
    "stsb",  # Publisher sentence pairs and labels only; no research analysis.
    "beir",  # Original retrieval assets and relations, no research analysis.
    "__init__",
    "adapters",
    "assets",
    "bindings",
    "catalog",
    "consumer",
    "conversations",  # Original parent-link input restoration, no research analysis.
    "execution",
    "gpqa",  # Publisher archive extraction and original option permutation only.
    "identity",
    "input_views",
    "locomo",  # Original conversation/question relations, no memory generation.
    "mapping_contract",
    "pipeline",
    "provider",
    "quality",  # Original article/question relation expansion only, no scoring.
    "qasper",  # Original paper/question/reference relations only, no analysis.
    "records",
    "registry",
    "taxonomy",
    "xlsum",  # Exact publisher test extraction/schema validation, no analysis.
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


def test_gpqa_helper_contains_only_original_data_preparation_dependencies():
    import ast

    tree = ast.parse((ROOT / "src/vllm_hust_workload_data/gpqa.py").read_text("utf-8"))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports == {
        "csv",
        "hashlib",
        "io",
        "json",
        "pathlib",
        "random",
        "zipfile",
        "registry",
    }


def test_quality_expansion_remains_pure_data_only():
    import ast

    tree = ast.parse(
        (ROOT / "src/vllm_hust_workload_data/quality.py").read_text(encoding="utf-8")
    )
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imports == {"copy"}
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))


def test_locomo_expansion_remains_pure_data_only():
    import ast

    tree = ast.parse(
        (ROOT / "src/vllm_hust_workload_data/locomo.py").read_text(encoding="utf-8")
    )
    assert {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } == {"copy"}
    assert {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } == {"re"}
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"open", "read_text", "write_text", "post", "request"}
        for node in ast.walk(tree)
    )
