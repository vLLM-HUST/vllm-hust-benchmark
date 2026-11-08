"""Create a first-time fixed-reference speaker bundle using one NPU control.

Default is plan-only. Explicit --execute requires one allocated visible NPU.
No previous qualification receipt is needed; only successful fixed input/output
and resource release produce a serving profile. Never executes a CPU network.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from full_duplex_npu_speaker import (
    DERIVED_GRAPH_SHA256,
    INPUT_SHA256,
    ORIGINAL_ONNX_SHA256,
    NpuSpeakerSession,
    digest,
)


def reference_features(path):
    import numpy as np
    from torchaudio.compliance import kaldi
    from full_duplex_reference_audio import load_reference_wave

    wave, rate = load_reference_wave(path)
    if rate != 16000:
        raise ValueError("original 16000Hz reference required")
    features = kaldi.fbank(wave[:1], num_mel_bins=80, dither=0, sample_frequency=16000)
    features = features - features.mean(dim=0, keepdim=True)
    data = np.ascontiguousarray(features.unsqueeze(0).numpy(), dtype=np.float32)
    if (
        list(data.shape) != [1, 1682, 80]
        or hashlib.sha256(data.tobytes()).hexdigest() != INPUT_SHA256
    ):
        raise ValueError("fixed original reference features differ")
    return data


def qualify(args):
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-9]+", visible) is None:
        raise ValueError("explicit single allocated ASCEND_RT_VISIBLE_DEVICES required")
    repository = Path(__file__).resolve().parents[1]
    for path in (
        args.output,
        args.original_model,
        args.derived_graph,
        args.model_om,
        args.reference_audio,
    ):
        if path.resolve().is_relative_to(repository):
            raise ValueError(
                "model artifacts and bundle must remain outside repository"
            )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "neural_execution": False,
        "benchmark_or_task_execution": False,
        "native_cases": 0,
        "source_checkbox_eligible": False,
        "full_network_numeric_parity_verified": False,
        "visible_physical_npu": visible,
    }
    session = None
    try:
        data = reference_features(args.reference_audio)
        model = args.output / "campplus.om"
        graph = args.output / "campplus.reference-shape.pool-lowered.onnx"
        shutil.copy2(args.model_om, model)
        shutil.copy2(args.derived_graph, graph)
        if (
            digest(args.original_model) != ORIGINAL_ONNX_SHA256
            or digest(graph) != DERIVED_GRAPH_SHA256
        ):
            raise ValueError(
                "original and derived speaker identities required before device use"
            )
        import torch
        import torch_npu  # noqa: F401 - registers the backend

        torch.npu.set_device(0)
        session = NpuSpeakerSession.for_reference_qualification(
            args.original_model, model, graph
        )
        report["neural_execution_attempted"] = True
        result = session.run(None, {"input": data})[0]
        report["neural_execution"] = True
        session.close()
        session = None
        report.update(
            passed=True,
            execution_device="NPU",
            input_shape=list(data.shape),
            input_features_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
            output_shape=list(result.shape),
            output_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
            model_om_sha256=digest(model),
            derived_graph_sha256=digest(graph),
            first_time_without_previous_receipt=True,
            resources_released=True,
        )
        profile = {
            "schema_version": "full-duplex-npu-speaker-profile/v1",
            "model_om": model.name,
            "derived_graph": graph.name,
            "qualification": "qualification-public.json",
        }
        (args.output / "profile.json").write_text(
            json.dumps(profile, indent=2), encoding="utf-8"
        )
    except Exception as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        active = sys.exception()
        try:
            if session is not None:
                session.close()
        except Exception as cleanup_error:
            report.update(passed=False, cleanup_error=str(cleanup_error))
            if active is not None:
                active.add_note("speaker qualification cleanup: " + str(cleanup_error))
            else:
                raise
        finally:
            (args.output / "qualification-public.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "original-model",
        "derived-graph",
        "model-om",
        "reference-audio",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_DEVICE_USE",
                    "neural_execution": False,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return 0
    qualify(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
