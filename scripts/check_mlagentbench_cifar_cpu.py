"""Probe a task-internal CPU environment; never run an Agent or training task."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/mlagentbench-cifar10-original-assets.json"
)


def validate_metadata(expected, observed):
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"qualified task CPU base differs: {key}")


def probe(args):
    profile = json.loads(args.profile.read_text("utf-8"))
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_NEURAL_IMPORT",
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        return report
    expected = profile["software"]["observed_candidate_base"]
    observed = {"python": platform.python_version(), "architecture": platform.machine()}
    observed.update(
        {name: metadata.version(name) for name in expected if name not in observed}
    )
    validate_metadata(expected, observed)
    if metadata.version("torchvision").split("+")[0] != "0.18.0":
        raise ValueError("exact qualified TorchVision release required")
    import torch
    import torchvision.datasets.cifar as cifar

    if torch.cuda.is_available():
        raise ValueError(
            "this qualified task environment is CPU-only, not a CUDA baseline"
        )
    loader = Path(cifar.__file__)
    digest = hashlib.sha256(loader.read_bytes()).hexdigest()
    if digest != profile["data"]["checksum_source_sha256"]:
        raise ValueError(
            "installed original CIFAR loader differs from pinned official wheel"
        )
    if (
        cifar.CIFAR10.tgz_md5 != profile["data"]["md5"]
        or cifar.CIFAR10.url != profile["data"]["url"]
    ):
        raise ValueError("installed dataset origin/checksum differs")
    report.update(
        status="TASK_CPU_DEPENDENCIES_AND_ORIGINAL_LOADER_IMPORTED_NOT_NATIVE_EXECUTION",
        observed=observed,
        torchvision=metadata.version("torchvision"),
        original_loader_sha256=digest,
        original_loader_train_files=len(cifar.CIFAR10.train_list),
        original_loader_test_files=len(cifar.CIFAR10.test_list),
        dataset_opened=False,
        scientific_training_executed=False,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(probe(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
