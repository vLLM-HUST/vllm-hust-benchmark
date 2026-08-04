#!/usr/bin/env python3
"""Patch v0.18.0 vllm/benchmarks/datasets.py to load HF datasets from local cache.

This applies the same VisionArenaDataset.load_data() override that current main uses,
plus a similar fallback for InstructCoderDataset.
"""

import sys
from pathlib import Path

DATASETS_PY = Path("/tmp/vllm-v0180/vllm/benchmarks/datasets.py")


def patch_vision_arena(content: str) -> str:
    """Add load_data() override to VisionArenaDataset."""
    if "Load data from local parquet cache" in content:
        print("VisionArenaDataset already patched")
        return content

    old = 'class VisionArenaDataset(HuggingFaceDataset):\n    """\n    Vision Arena Dataset.\n    """\n\n    DEFAULT_OUTPUT_LEN = 128\n    SUPPORTED_DATASET_PATHS'

    new = '''class VisionArenaDataset(HuggingFaceDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128

    def load_data(self) -> None:
        """Load data from local parquet cache when HF Hub is unreachable."""
        import glob
        import os
        cache_root = os.environ.get(
            "HF_HUB_CACHE",
            os.path.expanduser("~/.cache/huggingface/hub"),
        )
        cache_dir_name = "datasets--" + self.hf_name.replace("/", "--")
        cache_pattern = os.path.join(
            cache_root, cache_dir_name, "snapshots", "*", "data", "*.parquet"
        )
        parquet_files = sorted(glob.glob(cache_pattern))
        if parquet_files:
            from datasets import load_dataset as _load_dataset
            self.data = _load_dataset(
                "parquet",
                data_files=parquet_files[:3],
                split="train",
            )
            if not getattr(self, "disable_shuffle", False):
                self.data = self.data.shuffle(seed=self.random_seed)
            return
        super().load_data()

    SUPPORTED_DATASET_PATHS'''

    if old not in content:
        print("ERROR: VisionArenaDataset pattern not found")
        return content
    return content.replace(old, new)


def patch_instruct_coder(content: str) -> str:
    """Add load_data() override to InstructCoderDataset."""
    if "load_data_from_local_parquet" in content:
        print("InstructCoderDataset already patched")
        return content

    # Find InstructCoderDataset class and add load_data override after __init__
    old = 'class InstructCoderDataset(HuggingFaceDataset):\n    """\n    InstructCoder Dataset.\n    https://huggingface.co/datasets/likaixin/InstructCoder'

    new = '''class InstructCoderDataset(HuggingFaceDataset):
    """
    InstructCoder Dataset.
    https://huggingface.co/datasets/likaixin/InstructCoder

    Local parquet fallback added for offline benchmark environments.
    """

    def load_data(self) -> None:
        """Load data from local parquet/cache when HF Hub is unreachable."""
        import glob
        import os
        cache_root = os.environ.get(
            "HF_HUB_CACHE",
            os.path.expanduser("~/.cache/huggingface/hub"),
        )
        cache_dir_name = "datasets--" + self.hf_name.replace("/", "--")
        cache_pattern = os.path.join(
            cache_root, cache_dir_name, "snapshots", "*", "*.parquet"
        )
        parquet_files = sorted(glob.glob(cache_pattern))
        if parquet_files:
            from datasets import load_dataset as _load_dataset
            self.data = _load_dataset(
                "parquet",
                data_files=parquet_files,
                split="train",
            )
            if not getattr(self, "disable_shuffle", False):
                self.data = self.data.shuffle(seed=self.random_seed)
            return
        # Also check for arrow cache (datasets library cache format)
        datasets_cache = os.environ.get(
            "HF_DATASETS_CACHE",
            os.path.expanduser("~/.cache/huggingface/datasets"),
        )
        arrow_pattern = os.path.join(datasets_cache, "*", "*", "*", "*.arrow")
        arrow_files = sorted(glob.glob(arrow_pattern))
        if arrow_files:
            from datasets import load_from_disk
            try:
                cache_dir = os.path.dirname(os.path.dirname(arrow_files[0]))
                self.data = load_from_disk(cache_dir).get(
                    self.dataset_split,
                    load_from_disk(cache_dir).get("train"),
                )
                if not getattr(self, "disable_shuffle", False):
                    self.data = self.data.shuffle(seed=self.random_seed)
                return
            except Exception:
                pass
        super().load_data()

    # Original docstring continues below
    """InstructCoder is the dataset designed for general code editing.  It consists'''

    if old not in content:
        print("ERROR: InstructCoderDataset pattern not found")
        return content

    # Replace the class declaration and docstring start
    content = content.replace(old, new)

    # Remove the remaining original docstring lines that are now duplicated
    # The original has:
    #   InstructCoder is the dataset designed for general code editing.  It consists
    #   of ~500k ...
    # Our patch already included the first line, so we need to remove the duplicate
    content = content.replace(
        '    """InstructCoder is the dataset designed for general code editing.  It consists\n    of ~500k',
        "    InstructCoder is the dataset designed for general code editing.  It consists\n    of ~500k",
    )
    return content


def main():
    if not DATASETS_PY.exists():
        print(f"ERROR: {DATASETS_PY} not found")
        sys.exit(1)

    content = DATASETS_PY.read_text()
    original = content

    content = patch_vision_arena(content)
    content = patch_instruct_coder(content)

    if content == original:
        print("No changes made")
        sys.exit(0)

    # Backup
    backup = DATASETS_PY.with_suffix(".py.bak")
    backup.write_text(original)
    print(f"Backup saved to {backup}")

    DATASETS_PY.write_text(content)
    print(f"Patched {DATASETS_PY}")

    # Also patch the installed package if it exists
    installed = Path(
        "/root/miniconda3/envs/vllm-ascend-official-v0180/lib/python3.11/site-packages/vllm/benchmarks/datasets.py"
    )
    if installed.exists():
        installed_content = installed.read_text()
        installed_content = patch_vision_arena(installed_content)
        installed_content = patch_instruct_coder(installed_content)
        if installed_content != installed.read_text():
            installed_backup = installed.with_suffix(".py.bak")
            installed_backup.write_text(installed.read_text())
            installed.write_text(installed_content)
            print(f"Also patched installed package: {installed}")


if __name__ == "__main__":
    main()
