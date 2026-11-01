"""Prepare a fresh original Guo workspace with disclosed NPU/path-only ports."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_jsonschemabench_author_source import verify_source
from run_dailytalk_guo_encoder_npu import load_inputs, verify_saved
from verify_dailytalk_original_models import PROFILE, verify_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = ROOT / "configs/workload-native/dailytalk-author-source.json"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("pinned original compatibility patch context differs")
    return text.replace(old, new, 1)


def port_source(relative, raw):
    """Only the two previously qualified import/device compatibility changes."""
    if relative not in {"model/speaker_embedder.py", "synthesize.py"}:
        return raw
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if relative == "model/speaker_embedder.py":
        text = replace_once(text, "from deepspeaker import embedding\n", "")
        if text.count('        if self.embedder_type == "DeepSpeaker":\n') != 2:
            raise ValueError("pinned original optional speaker branch differs")
        text = text.replace(
            '        if self.embedder_type == "DeepSpeaker":\n',
            '        if self.embedder_type == "DeepSpeaker":\n'
            "            from deepspeaker import embedding\n",
        )
    else:
        text = replace_once(text, "from g2p_en import G2p\n", "")
        text = replace_once(
            text, "    g2p = G2p()", "    from g2p_en import G2p\n\n    g2p = G2p()"
        )
        old = (
            "    if torch.cuda.is_available():\n"
            '        torch.cuda.manual_seed(train_config["seed"])\n'
            "        device = torch.device('cuda')\n"
            "    else:\n"
            "        device = torch.device('cpu')"
        )
        new = (
            "    import torch_npu\n"
            "    if not torch.npu.is_available():\n"
            '        raise RuntimeError("NPU required; no CPU/CUDA neural fallback")\n'
            '    torch.npu.manual_seed_all(train_config["seed"])\n'
            '    device = torch.device(os.environ["DAILYTALK_NPU_DEVICE"])'
        )
        text = replace_once(text, old, new)
    return text.encode("utf-8")


def build_files(args, pins):
    import yaml

    generated = {}
    for item in pins["files"]:
        relative = item["path"]
        raw = (args.author_root / relative).read_bytes()
        if relative == "config/DailyTalk/preprocess.yaml":
            config = yaml.safe_load(raw)
            if config["preprocessing"]["speaker_embedder"] != "none":
                raise ValueError("original no-DeepSpeaker Guo configuration required")
            config["path"].update(
                corpus_path=str(args.audio_root.resolve()),
                raw_path=str(args.audio_root.resolve()),
                lexicon_path=str(
                    args.metadata_root.resolve() / "lexicon/librispeech-lexicon.txt"
                ),
                preprocessed_path=str(
                    args.prepared.resolve() / "preprocessed_data/DailyTalk"
                ),
            )
            raw = yaml.safe_dump(config, sort_keys=False).encode()
        elif relative == "config/DailyTalk/train.yaml":
            config = yaml.safe_load(raw)
            config["path"].update(
                ckpt_path=str(args.models_root.resolve()),
                log_path=str(args.stage.resolve() / "logs"),
                result_path=str(args.stage.resolve() / "results"),
            )
            raw = yaml.safe_dump(config, sort_keys=False).encode()
        generated[relative] = port_source(relative, raw)
    return generated


def prepare(args):
    report = {
        "source_id": "DAILYTALK",
        "status": "PLAN_ONLY_NO_MODEL_LOAD",
        "model_calls": 0,
        "native_synthesis_executed": False,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    verify_source(args.author_root, SOURCE_PROFILE)
    rows, contract = load_inputs(args)
    from types import SimpleNamespace

    verify_saved(
        SimpleNamespace(prepared=args.prepared, output=args.encoder_output),
        rows,
        contract,
    )
    profile = json.loads(PROFILE.read_text("utf-8"))
    for role, name in (
        ("acoustic", "900000.pth.tar"),
        ("vocoder", "generator_universal.pth.tar"),
    ):
        verify_file(args.models_root / name, profile[role])
    pins = json.loads(SOURCE_PROFILE.read_text("utf-8"))
    stage = args.stage.resolve()
    if stage.is_relative_to(ROOT) or any(
        stage.is_relative_to(path.resolve())
        for path in (
            args.author_root,
            args.prepared,
            args.models_root,
            args.metadata_root,
            args.audio_root,
        )
    ):
        raise ValueError("fresh external workspace must not overlap original assets")
    generated = build_files(args, pins)
    manifest = {
        "source_id": "DAILYTALK",
        "status": "ORIGINAL_GUO_NPU_WORKSPACE_PREPARED_NOT_EXECUTED",
        "publisher_revision": pins["revision"],
        "files": [
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in generated.items()
        ],
        "disclosed_source_ports": ["model/speaker_embedder.py", "synthesize.py"],
        "path_only_config_bindings": [
            "config/DailyTalk/preprocess.yaml",
            "config/DailyTalk/train.yaml",
        ],
        "original_utterances": len(rows),
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "prepare":
        stage.mkdir(mode=0o700, parents=True, exist_ok=False)
        workspace = stage / "workspace"
        for name, raw in generated.items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
        (workspace / "hifigan").mkdir(exist_ok=True)
        (workspace / "hifigan/generator_universal.pth.tar").symlink_to(
            (args.models_root / "generator_universal.pth.tar").resolve()
        )
        with (stage / "workspace-manifest.private.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2)
    saved = json.loads((stage / "workspace-manifest.private.json").read_text("utf-8"))
    if saved != manifest:
        raise ValueError("prepared workspace manifest differs from original bindings")
    for item in manifest["files"]:
        target = stage / "workspace" / item["path"]
        if target.is_symlink() or not target.resolve(strict=True).is_relative_to(stage):
            raise ValueError("prepared publisher file escapes its workspace")
        verify_file(target, item)
    expected_python = {name for name in generated if name.endswith(".py")}
    actual_python = {
        path.relative_to(stage / "workspace").as_posix()
        for path in (stage / "workspace").rglob("*.py")
    }
    if actual_python != expected_python:
        raise ValueError("unexpected or missing prepared publisher Python files")
    link = stage / "workspace/hifigan/generator_universal.pth.tar"
    if (
        not link.is_symlink()
        or link.resolve(strict=True)
        != (args.models_root / "generator_universal.pth.tar").resolve()
    ):
        raise ValueError("workspace vocoder must resolve to verified original weight")
    report.update(
        status="ORIGINAL_GUO_NPU_WORKSPACE_VERIFIED_NOT_EXECUTED",
        original_utterances=len(rows),
        software_files=len(generated),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "author-root",
        "prepared",
        "metadata-root",
        "models-root",
        "audio-root",
        "encoder-output",
        "stage",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
