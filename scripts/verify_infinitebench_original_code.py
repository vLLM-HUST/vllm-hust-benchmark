"""Read-only official code preflight, no imports, data downloads or model calls."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/infinitebench-author-code-pins.json"
)


def verify(checkout, pins=None):
    pins = json.loads(PINS.read_text("utf-8")) if pins is None else pins
    checkout = Path(checkout).resolve()
    verified = []
    for item in pins["files"]:
        path = (checkout / item["path"]).resolve()
        if not path.is_relative_to(checkout) or not path.is_file():
            raise ValueError("missing or escaped official code file")
        raw = path.read_bytes()
        digest = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if digest != item["git_blob"] or len(raw) != item["bytes"]:
            raise ValueError("official code identity mismatch")
        verified.append(item["path"])
    if "src/eval_utils.py" not in verified or "src/compute_scores.py" not in verified:
        raise ValueError("official prompts and scorer required")
    tree = ast.parse((checkout / "src/eval_utils.py").read_text("utf-8"))
    tables = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "DATA_NAME_TO_PATH",
                    "DATA_NAME_TO_MAX_NEW_TOKENS",
                }:
                    tables[target.id] = ast.literal_eval(node.value)
    paths = tables.get("DATA_NAME_TO_PATH", {})
    budgets = tables.get("DATA_NAME_TO_MAX_NEW_TOKENS", {})
    if (
        not paths
        or set(paths) != set(budgets)
        or any(type(value) is not int or value <= 0 for value in budgets.values())
    ):
        raise ValueError("official task/budget contract incomplete")
    return {
        "status": "OFFICIAL_CODE_STATIC_PREFLIGHT_ONLY",
        "revision": pins["revision"],
        "verified_files": verified,
        "tasks": [
            {"task": name, "file": filename, "max_new_tokens": budgets[name]}
            for name, filename in paths.items()
        ],
        "model_calls": 0,
        "publisher_code_executed": False,
        "dependencies_verified": False,
        "native_execution_complete": False,
        "source_checkbox_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.official_checkout), indent=2))
