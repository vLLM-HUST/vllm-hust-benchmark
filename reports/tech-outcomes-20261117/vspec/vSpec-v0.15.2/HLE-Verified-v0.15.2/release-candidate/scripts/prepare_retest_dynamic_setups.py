"""Prepare same-parameter vSpec setups with numerical proofs for a new NPU pair.

This is local evaluation plumbing. It changes only the paths of operator
proof directories; it does not change the plugin package or solver settings.
Run after validate_current_card_proofs.sh has completed on the allocated pair.
"""

import hashlib
import json
import subprocess
from pathlib import Path


WORK = Path("/root/recovery/issue18-delivery-20261113")
OLD_COMPONENTS = WORK / "components-current-card"
NEW_COMPONENTS = WORK / "components-retest-20261116"
OUT = WORK / "retest-20261116"
RUNNER = Path("/root/recovery/vspec-download-reinstall-20261108/examples/qwen35_three_datasets/paired.py")
CLIENT_PYTHON = Path("/root/recovery/vspec-client-clean-20261107/bin/python")
SERVER_PYTHON = Path("/root/recovery/vspec-install-r8-ZoYz6V/venv/bin/python")
BACKEND_PACKAGE = Path("/root/recovery/vspec-install-r8-ZoYz6V/venv/lib/python3.11/site-packages")


def write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def common_config(source: Path, expected_hash: str) -> dict:
    native = json.loads((source / "server-config/native.yaml").read_text(encoding="utf-8"))
    common = {key.replace("-", "_"): value for key, value in native.items()}
    for key in ("model", "host", "port", "served_model_name", "worker_extension_cls"):
        common.pop(key)
    common["speculative_config"].pop("num_speculative_tokens")
    digest = hashlib.sha256(json.dumps(common, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
    if digest != expected_hash:
        raise RuntimeError(f"Shared solver config changed: {source}")
    return common


def prepare(label: str, old_root: Path, new_root: Path) -> None:
    setup = json.loads((old_root / "setup.json").read_text(encoding="utf-8"))
    env = setup["dynamic_env"].copy()
    changed = []
    for key, value in env.items():
        if not isinstance(value, str) or not value.startswith(f"{OLD_COMPONENTS}/"):
            continue
        candidate = NEW_COMPONENTS / value.removeprefix(f"{OLD_COMPONENTS}/")
        if not candidate.is_dir():
            raise FileNotFoundError(f"New operator proof directory is absent: {candidate}")
        env[key] = str(candidate)
        changed.append(key)
    if len(changed) != 13:
        raise RuntimeError(f"Expected 13 changed operator proof paths, got {len(changed)}: {changed}")
    common_path = OUT / f"{label}-common.json"
    common_env_path = OUT / f"{label}-common-env.json"
    dynamic_env_path = OUT / f"{label}-dynamic-env.json"
    write_new(common_path, common_config(old_root, setup["common_config_sha256"]))
    write_new(common_env_path, setup["common_env"])
    write_new(dynamic_env_path, env)
    command = [
        str(CLIENT_PYTHON), str(RUNNER), "prepare",
        "--private-root", str(new_root),
        "--model-dir", setup["model_dir"],
        "--common-config", str(common_path),
        "--common-env", str(common_env_path),
        "--dynamic-env", str(dynamic_env_path),
        "--native-python", str(SERVER_PYTHON),
        "--dynamic-python", str(SERVER_PYTHON),
        "--client-python", str(CLIENT_PYTHON),
        "--native-backend-package", str(BACKEND_PACKAGE),
        "--dynamic-backend-package", str(BACKEND_PACKAGE),
        "--native-evaluation-dir", str(BACKEND_PACKAGE / "vllm_hust_vspec/evaluation"),
        "--dynamic-evaluation-dir", str(BACKEND_PACKAGE / "vllm_hust_vspec/evaluation"),
        "--served-model", setup["served_model"],
        "--port", str(setup["port"]),
        "--native-capacity", str(setup["native_capacity"]),
    ]
    subprocess.run(command, check=True)
    print(f"Prepared {label} vSpec retest setup with {len(changed)} fresh proof paths")


def main() -> None:
    OUT.mkdir(mode=0o700, exist_ok=True)
    prepare("fs", Path("/root/f1311bridge"), Path("/root/f1311bridge-retest-20261116"))
    prepare("hle", Path("/root/h1311hist1"), Path("/root/h1311hist1-retest-20261116"))


if __name__ == "__main__":
    main()
