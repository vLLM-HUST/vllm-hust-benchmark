"""Verify publisher acoustic/vocoder/history-encoder files without loading models.

Plan-only by default. Does not install libraries, download files, deserialize
checkpoints, initialize devices or qualify native speech synthesis.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/dailytalk-original-model-assets.json"


def verify_file(path, expected):
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"publisher model file size differs: {path.name}")
    sha = expected.get("sha256") or expected.get("lfs_sha256")
    digest = (
        hashlib.sha256()
        if sha
        else hashlib.sha1(b"blob " + str(expected["bytes"]).encode() + b"\0")
    )
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != (sha or expected["git_blob"]):
        raise ValueError(f"publisher model file digest differs: {path.name}")


def verify(args):
    profile = json.loads(PROFILE.read_text("utf-8"))
    report = {
        "source_id": "DAILYTALK",
        "status": "PLAN_ONLY_NO_MODEL_LOAD",
        "acoustic_variant": profile["acoustic"]["variant"],
        "restore_step": profile["acoustic"]["restore_step"],
        "publisher_history_window_turns": profile["acoustic"]["history_window_turns"],
        "vocoder_variant": profile["vocoder"]["variant"],
        "history_encoder": profile["encoder"]["model"],
        "encoder_embedding_dimension": profile["encoder"]["embedding_dimension"],
        "model_files_deserialized": False,
        "model_calls": 0,
        "npu_forward_qualified": False,
        "native_synthesis_or_scoring_executed": False,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    verify_file(args.acoustic_checkpoint, profile["acoustic"])
    verify_file(args.vocoder_checkpoint, profile["vocoder"])
    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_xsum_original_assets import bound_path

    for item in profile["encoder"]["files"]:
        verify_file(bound_path(args.encoder_root, "", item), item)
    report.update(
        status="ORIGINAL_MODEL_FILES_VERIFIED_NOT_NATIVE_EXECUTION",
        verified_files=2 + len(profile["encoder"]["files"]),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic-checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--action", choices=["plan", "verify"], default="plan")
    print(json.dumps(verify(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
