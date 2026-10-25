"""Audit built data-provider payload against current source, without installing.

This is a software artifact audit, not dataset, model, native execution or
research-consumer validation. Historical fixed-artifact receipts remain frozen.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "workload-preparation"
PACKAGE = "vllm_hust_workload_data"


def expected_payload(root):
    root = Path(root)
    result = {
        f"{PACKAGE}/{path.name}": path for path in (root / "src" / PACKAGE).glob("*.py")
    }
    for folder in ("data/catalogs", "schemas"):
        for path in (root / folder).glob("*.json"):
            key = f"{PACKAGE}/resources/{path.name}"
            if key in result:
                raise ValueError("ambiguous packaged resource basename")
            result[key] = path
    result[f"{PACKAGE}/resources/migration_provenance.json"] = (
        root / "MIGRATION_PROVENANCE.json"
    )
    return result


def audit(wheel, *, root=ROOT):
    expected = expected_payload(root)
    if not expected or not any(name.endswith("/__init__.py") for name in expected):
        raise ValueError("nonempty original provider source required")
    verified = []
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel members")
        actual = {
            name
            for name in names
            if name.startswith(PACKAGE + "/") and not name.endswith("/")
        }
        if actual != set(expected):
            raise ValueError(
                "wheel provider payload differs from current source inventory"
            )
        metadata_name = re.compile(
            r"vllm_hust_workload_data-[A-Za-z0-9_.+\-]+\.dist-info/(?:METADATA|WHEEL|top_level\.txt|RECORD|licenses/NOTICE\.md)"
        )
        if any(
            not name.startswith(PACKAGE + "/") and not metadata_name.fullmatch(name)
            for name in names
        ):
            raise ValueError(
                "unexpected payload outside provider and distribution metadata"
            )
        for name, source in expected.items():
            raw = bundle.read(name)
            if raw != source.read_bytes():
                raise ValueError("wheel payload bytes differ from reviewed source")
            verified.append(
                {
                    "path": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return {
        "status": "CURRENT_SOURCE_PAYLOAD_MATCHED_NOT_NATIVE_VALIDATION",
        "wheel_sha256": hashlib.sha256(Path(wheel).read_bytes()).hexdigest(),
        "verified_payloads": verified,
        "dataset_validation": False,
        "model_calls": 0,
        "installed_consumer_verified": False,
        "source_checkbox_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.wheel), indent=2))
