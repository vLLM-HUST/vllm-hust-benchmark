"""Offline static preflight; never imports publisher or executes candidate code."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from run_bigcodebench_original_generation import CODE_REVISION, DATA_SHA, digest

# Observed checksums of the fixed public source, not publisher signatures.
PINS = {
    "Docker/Evaluate.Dockerfile": "a0d56d681c9eeaf0c6d780309aab31b6df422bad21ad1a3de2c3da0598e05b12",  # pragma: allowlist secret - fixed public source checksum
    "Requirements/requirements.txt": "ac7c40cd926e1e949ebe241e4ba2f4ed1928f634d2dce32ac363e14c04cef807",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/__init__.py": "da556cf4bc44f5922a7b610aaa9e875111f4bb670a7ce8d2755b45fad532c9e0",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/data/__init__.py": "9359915bef845f7e2b1569f030a3f4acfe890e9da3744a38c1ea3cfec5e15a96",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/data/utils.py": "1d7d43e4dbfbbb8ca0dde4da73bfe3660c823e0ecad26d9d6707cc475bb2cb4f",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/eval/_special_oracle.py": "0cf930163987d30f455547aec6cbc500a154bc58a2eb109aa247ef4e0962448b",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/eval/utils.py": "9061f74fe937c4eb7a1b2bc423f7acab547ae01804d5e25933e2aa8a3cc2d685",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/gen/__init__.py": "7952440f3bd47b4f97f628b3d529dab58e57c42766f9e0594b4d4ccf55882dbb",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/gen/util/__init__.py": "759e129e4115ec596c9cd8784f7d37ca909b2cefc02afccf4eb89837d59cf249",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/gen/util/anthropic_request.py": "bf1dc48d128fd86528367bef8adb276e3e65dfd6dcc16024c8bc507eb985eae1",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/gen/util/google_request.py": "050272723b66c6ba1c6fe04cdf7b8db3acadf644ab240953b7c9a9cbd5bfad0b",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/gen/util/hf_inference_request.py": "0d65645819069cfd7ac5e0dd1e36da08a3b65ece839d63a5c06f7f5f7f9dbe52",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/gen/util/mistral_request.py": "20afed09257393d6536ed16f3084c7011db2b4e8c35898558108dfa953b81f88",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/gen/util/openai_request.py": "f7d147d3ee01dc77db9c45f1b567b76614331d8ddf9b46db0cfc39081648457e",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/generate.py": "12325ca6da7a6469bc76084b0030762065036b66ffa69c2e6211c03bd6075203",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/inspect.py": "8d958c3ee75912ad13a3180731b2bfe0cc0d481e5c0f5c8fc829edd671145243",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/__init__.py": "eb645f300b7666fc2bd128a85e46e8fd5d40f9fb77e3a4f80fd4e6decbd7cdbd",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/anthropic.py": "0e6c47b728562a731d1300ac49d9a9e78e590c2cd1a67113114f4ab725aff782",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/base.py": "dfd2ab4dc584592d48c093557dfa5f49c2c2a6821f3f36a091f5e43043fa3f85",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/google.py": "d357d524e54b86d53ba27065912f004fcdc38f38bc291d186a6082fbbeb2fdb2",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/hf.py": "5ae95dd1409976d48b0439122ba6f0cea720c4431c5ccb83d2414aed3a06f7b8",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/hf_inference.py": "92629b32ff096fcd66a86096b963f118c89b055e4193eeb3f0d1267bbd7050af",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/mistral.py": "a108bdea8d5baec457208d6475e5df6228550d9b4a6c19f2fa4de5bd7b336fa6",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/openai.py": "c2a128aef5b3bc4b3fec25f299603b82689cf3697e29eb669c6fbe1a33bd5559",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/utility.py": "a58690bedcdcb0bac9062bcef94c0ecbe1ef204e7fb7bebf6a6423e6a5900eaf",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/provider/vllm.py": "0805bc48341a4d3518647b71aa6b1741646d6754f9c068d9b884ffce02e5503a",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/syncheck.py": "7545ffb2eedf0035067898170c09f01637cd203e6204b88e34bea0ecc7b39895",  # pragma: allowlist secret - fixed public source checksum
    "bigcodebench/evaluate.py": "6f737ddbec85c8e24d995898bab528b8c28fc5090a488652c8af557ad85f0667",  # pragma: allowlist secret
    "bigcodebench/sanitize.py": "c3b34cd425b0eb62afaf84f4268af787d317ce552685dd6d9b396442c91f47e4",  # pragma: allowlist secret
    "bigcodebench/eval/__init__.py": "d5fd553559ac1b76659ebc32ae30e3e779449ecd31c5201f2301319ceeee01fe",  # pragma: allowlist secret
    "bigcodebench/data/bigcodebench.py": "cac3e3a8425a1fc56816caf6133c193866aadd729c0bb735d8a00a387178007a",  # pragma: allowlist secret
    "Requirements/requirements-eval.txt": "a4d01fb12cbce5223b51f982265cb7975bea770b758cd85cc91b803d3293e39f",  # pragma: allowlist secret
}


def verify_source(root, pins=PINS):
    root = Path(root).resolve(strict=True)
    if pins is PINS:
        actual = {
            path.relative_to(root).as_posix()
            for path in (root / "bigcodebench").rglob("*.py")
        }
        expected = {name for name in pins if name.endswith(".py")}
        if actual != expected:
            raise ValueError(
                "complete fixed Python source tree required; no extra modules"
            )
    verified = {}
    for name, expected in pins.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("publisher source path escapes root")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"publisher checksum mismatch: {name}")
        verified[name] = observed
    return verified


def original_imports(row):
    """Reference bodies are indented continuations of complete_prompt."""
    modules = set()
    for text in (row["complete_prompt"] + row["canonical_solution"], row["test"]):
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                modules.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError(
                        "relative import requires explicit environment review"
                    )
                modules.add((node.module or "").split(".")[0])
    return sorted(modules)


def audit(root, rows):
    checksums = verify_source(root)
    if len(rows) != 1140 or [r["task_id"] for r in rows] != [
        f"BigCodeBench/{i}" for i in range(1140)
    ]:
        raise ValueError("full original ordered 1140 tasks required")
    return {
        "source_id": "BIGCODEBENCH",
        "status": "STATIC_SCORING_ASSETS_VERIFIED_NOT_EXECUTED",
        "code_revision": CODE_REVISION,
        "publisher_source_checksums": checksums,
        "environment_preparation": {
            "publisher_recipe": "Docker/Evaluate.Dockerfile",
            "publisher_cli_dependencies": "Requirements/requirements.txt",
            "publisher_scoring_dependencies": "Requirements/requirements-eval.txt",
            "recipe_is_fully_pinned": False,
            "floating_inputs_requiring_freeze": [
                "python:3.10-slim base image",
                "apt repositories and packages",
                "latest upstream git clone",
                "main branch requirements-eval URL",
                "unversioned pip dependency installs",
                "build-time full/hard dataset downloads",
            ],
            "required_build_record": [
                "resolved base-image digest and architecture",
                "fixed publisher source and requirements checksums",
                "resolved OS and Python package versions",
                "immutable final image ID",
            ],
            "image_built_or_qualified": False,
            "boundary": "Pinned recipe identity is not reproducible image qualification. Do not execute its floating clone/dependency/data acquisition unchanged.",
        },
        "selected_original_indices": [0, 1, 2],
        "selected_task_static_imports": [
            {"task_id": row["task_id"], "imports": original_imports(row)}
            for row in rows[:3]
        ],
        "static_import_limit": "Not exhaustive: dynamic imports, runtime files and native dependencies still require actual isolated execution and reference checks.",
        "candidate_code_executed": False,
        "official_sanitize_executed": False,
        "official_functional_scoring_executed": False,
        "environment_qualified": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    path = args.data_root / "data/v0.1.4-00000-of-00001.parquet"
    if digest(path) != DATA_SHA:
        raise ValueError("original publisher dataset identity mismatch")
    rows = pq.read_table(path).to_pylist()
    if digest(path) != DATA_SHA:
        raise ValueError("original publisher dataset changed during reading")
    result = audit(args.official_code, rows)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
