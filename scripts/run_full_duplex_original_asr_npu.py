"""Original reference Parakeet ASR with explicit NPU placement.

Default is a plan with no imports of neural runtimes. Execute in the separately
qualified ASR environment, after native capture and original reference export.
This is a device binding, not a replacement transcription or scoring algorithm.
"""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys

CHECKPOINT_SHA256 = "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"  # pragma: allowlist secret - publisher checkpoint


def verify_checkpoint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("original pinned Parakeet checkpoint differs")


def require_npu_parameters(model):
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if devices != ["npu:0"]:
        raise ValueError("all neural parameters must be on npu:0: " + str(devices))
    return devices


def execute(args):
    verify_checkpoint(args.nemo)
    if args.output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("use an external output directory")
    if args.version == "v1.0" and (args.tree / "asr.json").exists():
        raise FileExistsError("reuse existing v1.0 ASR instead of overwriting it")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "source_id": "FULL_DUPLEX_BENCH",
        "version": args.version,
        "cpu_neural_fallback": False,
        "source_checkbox_eligible": False,
        "physical_device_selection": "inherited ASCEND_RT_VISIBLE_DEVICES",
        "original_asr_source_modified": False,
        "status": "STARTED",
    }
    sys.path.insert(0, str(args.omni_root))
    try:
        with (args.output / "asr.log").open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import torch
                import torch_npu  # noqa: F401 - registers the NPU backend
                from benchmarks.duplex.reference_asr import load_nemo_model
                from benchmarks.duplex.reference_source import verify_reference

                torch.npu.set_device(0)
                if args.version == "v1.0":
                    from benchmarks.duplex.reference_core import V10_REFERENCE_FILES
                    from benchmarks.duplex.reference_v10 import transcribe

                    paths = verify_reference(args.reference_source, V10_REFERENCE_FILES)
                    model = load_nemo_model(args.nemo, "npu:0")
                    report["parameter_devices"] = require_npu_parameters(model)
                    counts = transcribe(
                        args.tree,
                        paths,
                        args.nemo,
                        CHECKPOINT_SHA256,
                        "npu:0",
                        model=model,
                    )
                else:
                    from benchmarks.eval.benchmark_duplex_reference import open_engines
                    from benchmarks.duplex.reference_asr import run_asr
                    from benchmarks.duplex.reference_core import HashCache

                    native = argparse.Namespace(
                        phase="asr",
                        tree=[args.engine + "=" + str(args.tree)],
                        manifest=[],
                        manifest_projection=None,
                        only=[],
                        out=args.output / "scores",
                        reference_source=args.reference_source,
                        nemo=args.nemo,
                        nemo_sha256=CHECKPOINT_SHA256,
                        device="npu:0",
                        limit=None,
                        retry_failed=False,
                    )
                    paths = verify_reference(args.reference_source)
                    engines = open_engines(native)
                    native.out.mkdir()
                    hashes = HashCache(native.out / "hash-cache.json")
                    model = load_nemo_model(args.nemo, "npu:0")
                    report["parameter_devices"] = require_npu_parameters(model)
                    counts = run_asr(native, engines, paths, hashes, model=model)
                    hashes.save()
                report["counts"] = dict(counts)
                if not counts or not sum(counts.values()) or counts.get("failed", 0):
                    raise RuntimeError(
                        "original ASR reported failed cases; preserve them"
                    )
                report["status"] = "ORIGINAL_ASR_COMPLETED_NOT_WHOLE_BENCHMARK"
    except (Exception, SystemExit) as error:
        report.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (args.output / "qualification-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1.0", "v1.5"], required=True)
    for flag in ["omni-root", "reference-source", "tree", "nemo", "output"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--engine", default="minicpmo-npu-derived")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name in ["omni_root", "reference_source", "tree", "nemo", "output"]:
        setattr(args, name, getattr(args, name).resolve())
    if not args.engine or "=" in args.engine:
        parser.error("engine must be a nonempty label without '='")
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_MODEL_CALLS",
                    "version": args.version,
                    "device": "npu:0",
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return 0
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
