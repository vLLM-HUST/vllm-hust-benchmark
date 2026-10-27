"""Two additional original NI instruction tasks; not full WSC or BTC benchmarks.

Reuse the byte-pinned standalone NI runner without changing its defaults, official
collator, scorer, token budgets or generation behavior. Only the fixed task-asset
selection differs in this separate process. All task data and outputs stay private.
"""

import hashlib
import importlib.util
from pathlib import Path

CORE_SHA = "4c46f9ea6e862d91771049424221e0c69b67eefbc610f64e25fb82dd6a3e87ae"  # pragma: allowlist secret - existing validated runner checksum
TASKS = {
    "task275_enhanced_wsc_paraphrase_generation.json": (
        "dc282472e53940817ad6a696b20cc54b485e99fa1a06b06cd9ea5cc72ac8b349",  # pragma: allowlist secret - fixed public task checksum
        1178,
    ),
    "task1453_person_entity_extraction_btc_corpus.json": (
        "6656b6bdb003ed61ed2e21a4dc960a0d5d37f9bb75f137c3120a2dbb1139c0ae",  # pragma: allowlist secret - fixed public task checksum
        261,
    ),
}


def load_core():
    path = Path(__file__).with_name("run_natural_instructions_original.py")
    if (
        hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != CORE_SHA
    ):
        raise ValueError("fixed NI runner identity mismatch")
    spec = importlib.util.spec_from_file_location("ni_editing_extraction_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    core = load_core()
    # Fixed input asset configuration only, not an upstream source patch or engine hook.
    original_tasks = core.TASKS
    try:
        core.TASKS = TASKS
        core.main()
    finally:
        core.TASKS = original_tasks


if __name__ == "__main__":
    main()
