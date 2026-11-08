"""Prepare a new source-only NPU compatibility copy; never edit upstream in place.

Checks pinned affected files, records every copied source hash, and prepares
only disclosed device/reference bindings. Does not install dependencies, load
models, execute benchmarks or qualify neural execution. Default is plan-only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_REVISION = "272845a9caadb82f220d7f3d0b64fdd1098fbbca"  # pragma: allowlist secret - publisher revision
PREFIX = "sglang_omni/models/minicpm_o/components/"
SPECS = {
    PREFIX + "code2wav.py": {
        "sha256": "c9d3c379624273d244f39d07849283adce194a3889fe967255d6a77b1a631591",  # pragma: allowlist secret - original source hash
        "replacements": [
            (
                'if resolved_device.type not in {"cuda", "xpu"}:',
                'if resolved_device.type not in {"cuda", "xpu", "npu"}:',
            ),
            (
                "Token2wav requires a CUDA or XPU device, got {device}",
                "Token2wav requires a CUDA, XPU or disclosed NPU backend, got {device}",
            ),
        ],
    },
    "sglang_omni/comm/router.py": {
        "sha256": "cfa62fe4ec250dfa01b31039a7324118382c45b53f0c6f8855aec63763244348",  # pragma: allowlist secret - original source hash
        "replacements": [
            (
                'if obj.device.type in ("cpu", "musa"):',
                'if obj.device.type in ("cpu", "musa", "npu"):',
            )
        ],
    },
    PREFIX + "token2wav/vocoder.py": {
        "sha256": "91b67f0dc20026fe7dd3589cc02bbb16ecf4cd0158dfc42ba3f51ab78e3d2314",  # pragma: allowlist secret - original source hash
        "replacements": [
            (
                "        self.speaker_model = onnxruntime.InferenceSession(\n"
                '            str(model_path / "campplus.onnx"),\n'
                "            sess_options=options,\n"
                '            providers=["CPUExecutionProvider"],\n'
                "        )",
                "        from full_duplex_npu_speaker import NpuSpeakerSession\n"
                '        self.speaker_model = NpuSpeakerSession(model_path / "campplus.onnx")',
            ),
            (
                "        audio, sample_rate = torchaudio.load(source)",
                "        from full_duplex_reference_audio import load_reference_wave\n"
                "        audio, sample_rate = load_reference_wave(source)",
            ),
        ],
    },
}
HELPERS = ("full_duplex_npu_speaker.py", "full_duplex_reference_audio.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    files = {}
    total = 0
    for package in ("sglang_omni", "benchmarks"):
        directory = root / package
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("original package directory required: " + package)
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("source copy refuses symlinks")
            if (
                "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo"}
                or not path.is_file()
            ):
                continue
            total += path.stat().st_size
            if total > 64 * 1024**2:
                raise ValueError("source-only copy exceeds byte bound")
            files[path.relative_to(root).as_posix()] = sha(path)
    for name in ("LICENSE", "NOTICE"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            files[name] = sha(path)
    for name, spec in SPECS.items():
        if files.get(name) != spec["sha256"]:
            raise ValueError("pinned original affected file differs: " + name)
    return files, total


def prepare(root, output, files):
    repository = Path(__file__).resolve().parents[1]
    if (
        output.is_relative_to(repository)
        or output.is_relative_to(root)
        or root.is_relative_to(output)
    ):
        raise ValueError("use a new external directory separate from upstream source")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "STARTED",
        "declared_upstream_revision": SOURCE_REVISION,
        "source_checkbox_eligible": False,
        "models_downloaded": False,
        "neural_execution": False,
        "original_source_modified": False,
    }
    try:
        destination = output / "source"
        for name, expected in files.items():
            source = root / name
            target = destination / name
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise ValueError("source path changed during compatibility copy")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha(target) != expected:
                raise ValueError("source changed during compatibility copy: " + name)
        changed = {}
        for name, spec in SPECS.items():
            path = destination / name
            text = path.read_text("utf-8")
            for original, derived in spec["replacements"]:
                if text.count(original) != 1:
                    raise ValueError("original patch anchor differs: " + name)
                text = text.replace(original, derived)
            compile(text, name, "exec")
            path.write_text(text, encoding="utf-8")
            changed[name] = {
                "original_sha256": spec["sha256"],
                "derived_sha256": sha(path),
            }
        helper_dir = output / "runtime-helpers"
        helper_dir.mkdir()
        for name in HELPERS:
            shutil.copy2(Path(__file__).resolve().parent / name, helper_dir / name)
        (output / "original-source-manifest.json").write_text(
            json.dumps(files, indent=2), encoding="utf-8"
        )
        report.update(
            status="SOURCE_COMPATIBILITY_COPY_PREPARED_NOT_EXECUTED",
            copied_source_files=len(files),
            changed_files=changed,
            runtime_helpers={name: sha(helper_dir / name) for name in HELPERS},
            other_copied_source_files_unchanged=True,
            whole_publisher_tree_identity_claimed=False,
        )
    except Exception as error:
        report.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (output / "qualification-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = args.omni_root.resolve(strict=True)
    files, size = inventory(root)
    if args.prepare:
        report = prepare(root, args.output.resolve(), files)
    else:
        report = {
            "status": "PLAN_ONLY_NO_COPY",
            "affected_original_files_verified": len(SPECS),
            "source_files": len(files),
            "source_bytes": size,
            "neural_execution": False,
            "source_checkbox_eligible": False,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
