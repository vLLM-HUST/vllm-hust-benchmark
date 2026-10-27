"""No dependency installation or optional backend imports."""

from importlib import metadata
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "schema_probe", ROOT / "scripts/probe_jsonschemabench_environment.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_matching_metadata_is_never_native_completion():
    versions = {**runner.CORE_REQUIREMENTS, **runner.WORKFLOW_REQUIREMENTS}
    report = runner.probe(versions.__getitem__)
    assert report["all_listed_author_versions_match"]
    assert not report["native_execution_complete"]
    assert not report["source_checkbox_eligible"]
    assert report["installs_or_model_calls"] == 0


def test_missing_dependency_and_version_difference_are_distinct():
    def version(name):
        if name == "dacite":
            raise metadata.PackageNotFoundError(name)
        return "software-fixture-version"

    report = runner.probe(version)
    rows = {row["distribution"]: row for row in report["core_interface_dependencies"]}
    assert rows["dacite"]["status"] == "MISSING"
    assert rows["numpy"]["status"] == "VERSION_DIFFERS_NOT_A_COMPATIBILITY_VERDICT"
    assert not report["core_packages_present"]
    assert not report["all_listed_author_versions_match"]
