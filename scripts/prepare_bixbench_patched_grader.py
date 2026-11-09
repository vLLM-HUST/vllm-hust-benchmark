"""Materialize an opt-in BixBench grader overlay; never edit publisher files."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "bixbench/graders.py"
PROFILE = "bixbench-local-empty-match-range-template-v1"


def patch_source(source: str) -> str:
    edits = (
        (
            "correct = cleaned_predicted in cleaned_target",
            "correct = bool(cleaned_predicted) and cleaned_predicted in cleaned_target",
        ),
        (
            'if self.evaluation_mode == "range_verifier":\n'
            "            return await grading_func._grade_range_llm_verifier(\n"
            "                question=question,\n"
            "                target=target,\n"
            "                predicted=predicted,\n"
            "                llm_client=self.llm_client,\n"
            "                grading_prompt_template=self.grading_prompt_template,\n",
            'if self.evaluation_mode == "range_verifier":\n'
            "            return await grading_func._grade_range_llm_verifier(\n"
            "                question=question,\n"
            "                target=target,\n"
            "                predicted=predicted,\n"
            "                llm_client=self.llm_client,\n"
            "                grading_prompt_template=(\n"
            "                    OPEN_ENDED_RANGE_GRADING_PROMPT\n"
            "                    if self.grading_prompt_template == OPEN_ENDED_GRADING_PROMPT\n"
            "                    else self.grading_prompt_template\n"
            "                ),\n",
        ),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("unexpected author grader layout; refusing patch")
        source = source.replace(old, new, 1)
    compile(source, SOURCE, "exec")
    return source


def prepare(author_root: Path, output: Path) -> dict:
    manifest = json.loads(
        (ROOT / "configs/workload-native/bixbench-author-source.json").read_text(
            encoding="utf-8"
        )
    )
    expected = next(entry for entry in manifest["files"] if entry["path"] == SOURCE)
    raw = (author_root / SOURCE).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("publisher grader SHA-256 mismatch")
    if output.resolve().is_relative_to(author_root.resolve()):
        raise ValueError("overlay must be outside the publisher checkout")
    patched = patch_source(raw.decode("utf-8")).encode("utf-8")
    output.mkdir(exist_ok=False)
    (output / "bixbench").mkdir()
    (output / SOURCE).write_bytes(patched)
    receipt = {
        "profile": PROFILE,
        "author_revision": manifest["revision"],
        "original_grader_sha256": digest,
        "patched_grader_sha256": hashlib.sha256(patched).hexdigest(),
        "official_unmodified_score": False,
        "scoring_executed": False,
    }
    (output / "overlay-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.author_root, args.output), indent=2))
