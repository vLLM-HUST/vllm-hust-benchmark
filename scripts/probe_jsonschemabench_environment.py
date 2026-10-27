"""Read installed metadata only; never install, import optional engines or use NPU.

Core-interface readiness is narrower than full author package/backend readiness.
Missing/different dependencies remain explicit, not a native-delivery result.
"""

import argparse
from importlib import metadata
import json

# Fixed author requirements, epfl-dlab/JSONSchemaBench at the documented revision.
CORE_REQUIREMENTS = {
    "numpy": "2.2.3",
    "dacite": "1.9.2",
    "omegaconf": "2.3.0",
    "matplotlib": "3.10.1",
    "prettytable": "3.15.1",
}
WORKFLOW_REQUIREMENTS = {
    "datasets": "3.3.2",
    "huggingface_hub": "0.29.2",
    "jsonschema": "4.23.0",
    "protobuf": "6.30.0",
    "requests": "2.32.3",
    "stopit": "1.1.2",
    "tiktoken": "0.9.0",
    "tokenizers": "0.21.0",
    "torch": "2.6.0",
    "tqdm": "4.67.1",
    "transformers": "4.49.0",
}


def probe(version=metadata.version):
    def check(requirements):
        rows = []
        for name, expected in requirements.items():
            try:
                actual = version(name)
            except metadata.PackageNotFoundError:
                actual = None
            rows.append(
                {
                    "distribution": name,
                    "author_version": expected,
                    "installed_version": actual,
                    "status": "MISSING"
                    if actual is None
                    else "AUTHOR_VERSION_MATCH"
                    if actual == expected
                    else "VERSION_DIFFERS_NOT_A_COMPATIBILITY_VERDICT",
                }
            )
        return rows

    core, workflow = check(CORE_REQUIREMENTS), check(WORKFLOW_REQUIREMENTS)
    return {
        "status": "DEPENDENCY_METADATA_ONLY_NOT_RUNTIME_VALIDATION",
        "author_requirements": "https://github.com/epfl-dlab/JSONSchemaBench/blob/8003e8405c4d8d8b327b1eb472c9856297d75493/requirements.txt",
        "core_interface_dependencies": core,
        "additional_author_workflow_dependencies": workflow,
        "core_packages_present": all(
            row["installed_version"] is not None for row in core
        ),
        "all_listed_author_versions_match": all(
            row["status"] == "AUTHOR_VERSION_MATCH" for row in core + workflow
        ),
        "author_requirements_are_complete": "NOT_ESTABLISHED; optional backends and OpenAI SDK require separate checks",
        "core_package_eager_imports_optional_engines": True,
        "native_execution_complete": False,
        "source_checkbox_eligible": False,
        "installs_or_model_calls": 0,
        "serving_environment_change": "PROHIBITED_BY_THIS_PROBE; use separate client environment, never replace live NPU torch stack to match CPU client requirements",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(probe(), indent=2))
