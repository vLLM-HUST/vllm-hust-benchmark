"""The migration must not import the research project's observation subsystem."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "workload-preparation"
DATA_MODULES = {
    "deepplanning",  # Original queries/metadata/database references only.
    "scienceworld",  # Original publisher task/variation/reference relations only.
    "docvqa",  # Original field types and media paths only; no inference or analysis.
    "copilot",  # Original release integrity and nested metadata only, no analysis.
    "crosscodeeval",  # Original archive/profile/task relations only; no analysis.
    "flores200",  # Original parallel text and metadata bindings, no scoring.
    "narrativeqa",  # Pinned original table relations only, no observation/scoring.
    "schemastore",  # Original schema/fixture bytes and identities only, no execution.
    "multinews",  # Original paired documents/references, no research analysis.
    "librispeech",  # Original audio/transcript bindings only, no observations.
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


def test_docvqa_adapter_remains_data_only():
    import ast

    tree = ast.parse(
        (ROOT / "src/vllm_hust_workload_data/docvqa.py").read_text("utf-8")
    )
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imports == {"pathlib"}
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))


def test_crosscodeeval_archive_reader_remains_data_only():
    import ast

    tree = ast.parse(
        (ROOT / "src/vllm_hust_workload_data/crosscodeeval.py").read_text("utf-8")
    )
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {"copy", "json", "pathlib", "tarfile"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"extract", "extractall"}
        for node in ast.walk(tree)
    )


def test_no_observation_dependencies_or_hooks():
    requirements = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert all(
        name not in requirements for name in ("openai", "httpx", "psutil", "torch")
    )
    assert not list((ROOT / "tools").rglob("*.py"))
    assert not (ROOT / "schemas/full_cycle_trace_v1.schema.json").exists()


def test_copilot_integrity_reader_cannot_invoke_models_network_or_research():
    import ast

    tree = ast.parse(
        (ROOT / "src/vllm_hust_workload_data/copilot.py").read_text("utf-8")
    )
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {"datetime", "gzip", "hashlib", "json", "pathlib", "tarfile"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"extract", "extractall"}
        for node in ast.walk(tree)
    )


def test_schemastore_reader_cannot_execute_extract_download_or_observe():
    import ast

    tree = ast.parse(
        (ROOT / "src/vllm_hust_workload_data/schemastore.py").read_text("utf-8")
    )
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {"copy", "hashlib", "pathlib", "zipfile", "records", "assets"}
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"extract", "extractall", "urlopen", "system", "Popen"}
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"eval", "exec", "compile"}
        for node in ast.walk(tree)
    )


def test_new_native_entry_software_contracts_are_watched_and_cpu_smoked():
    import fnmatch
    import re

    repo = ROOT.parent
    text = (repo / ".github/workflows/workload-data-smoke.yml").read_text("utf-8")
    # Check this controlled workflow's text contracts without a YAML dependency.
    # General YAML syntax remains covered by the repository's lint hooks.
    paths = text.split("    paths:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
    watched = re.findall(r"""^\s+- ['"]([^'"]+)['"]\s*$""", paths, re.MULTILINE)
    scripts = [
        "run_historical_dialogue_original.py",
        "score_qrecc_original_qa.py",
        "prepare_mbpp_original_assets.py",
        "run_mbpp_original_generation.py",
        "prepare_mbpp_scoring_inputs.py",
        "run_mbpp_isolated_scoring.py",
        "mbpp_isolated_worker.py",
        "prepare_mooncake_storage_plan.py",
        "run_mooncake_storage_original.py",
        "audit_toolace_original_corpus.py",
        "humaneval_isolated_worker.py",
        "run_humaneval_isolated_scoring.py",
    ]
    fixtures = [
        "test_historical_dialogue_prefix.py",
        "test_historical_dialogue_runner.py",
        "test_qrecc_original_qa_score.py",
        "test_mbpp_original_assets.py",
        "test_mbpp_original_generation.py",
        "test_mbpp_scoring_preparation.py",
        "test_mbpp_isolated_scoring.py",
        "test_mooncake_storage_plan.py",
        "test_toolace_original_corpus_audit.py",
    ]
    for path in [
        *("scripts/" + name for name in scripts),
        *("tests/" + name for name in fixtures),
    ]:
        assert (repo / path).is_file()
        assert any(fnmatch.fnmatchcase(path, pattern) for pattern in watched), path
    assert "    runs-on: ubuntu-latest\n" in text
    commands = [
        line.strip().removeprefix("run: ")
        for line in text.splitlines()
        if line.strip().startswith(("run: python", "python "))
    ]
    for name in fixtures:
        assert any(
            "python -m pytest" in command and "tests/" + name in command
            for command in commands
        ), name
    assert not any(command.startswith("python scripts/run_") for command in commands)
    marker = (
        "      - name: Verify installed wheel payload outside checkout, no datasets\n"
    )
    assert text.count(marker) == 1
    installed = text.split(marker, 1)[1].split("      - name:", 1)[0]
    assert "working-directory: ${{ runner.temp }}" in installed
    assert "installed_wheel_audit" in installed
    assert "INSTALLED_PAYLOAD_AUDIT.json" in installed


def test_narrativeqa_relations_remain_data_only():
    import ast

    tree = ast.parse(
        (ROOT / "src/vllm_hust_workload_data/narrativeqa.py").read_text("utf-8")
    )
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imports == {"copy", "assets", "registry"}


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
