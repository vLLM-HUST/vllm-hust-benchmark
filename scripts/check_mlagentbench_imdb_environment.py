"""Check the IMDb task's isolated client; never run Agent actions or training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mlagentbench_local_agent import check_isolation, verify_port

TOKENIZER_FILE = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"  # pragma: allowlist secret - public tiktoken cache filename
TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"  # pragma: allowlist secret - public tokenizer digest


def append_paths(paths):
    """Fill missing packages without shadowing installed base dependencies."""
    for path in paths:
        resolved = str(Path(path).resolve(strict=True))
        if not Path(resolved).is_dir():
            raise ValueError("dependency directory required")
        if resolved not in sys.path:
            sys.path.append(resolved)


def probe(args):
    if not Path("/.dockerenv").is_file():
        raise ValueError("dedicated isolated container required")
    check_isolation(Path("/proc/self/status").read_text())
    cache = args.tokenizer_cache.resolve(strict=True)
    if (
        hashlib.sha256((cache / TOKENIZER_FILE).read_bytes()).hexdigest()
        != TOKENIZER_SHA256
    ):
        raise ValueError("original tokenizer cache identity differs")
    append_paths([args.compatibility, args.supplemental, args.port_root])
    verify_port(args.port_root)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import MLAgentBench.runner  # noqa: F401
    import accelerate
    import datasets
    import huggingface_hub
    import numpy
    import pyarrow
    from transformers import DistilBertForSequenceClassification, Trainer  # noqa: F401

    result = {
        "status": "SOFTWARE_AND_DATA_PROBE_NOT_NATIVE_TASK_EXECUTION",
        "versions": {
            "numpy": numpy.__version__,
            "huggingface-hub": huggingface_hub.__version__,
            "pyarrow": pyarrow.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
        },
        "model_calls": 0,
        "agent_actions": 0,
        "npu_training_qualified": False,
        "source_checkbox_eligible": False,
    }
    if args.imdb_root is not None:
        # Match the original short-name call via a read-only ./imdb mount.
        if (Path.cwd() / "imdb").resolve(strict=True) != args.imdb_root.resolve(
            strict=True
        ):
            raise ValueError("original assets must be mounted as ./imdb")
        data = datasets.load_dataset("imdb")
        rows = {key: len(value) for key, value in data.items()}
        if rows.get("train") != 25000 or rows.get("test") != 25000:
            raise ValueError("original IMDb train/test split sizes differ")
        result["original_split_rows"] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["compatibility", "supplemental", "port-root", "tokenizer-cache"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--imdb-root", type=Path)
    print(json.dumps(probe(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
