"""Prepare the disclosed fixed-reference speaker graph; never execute a network.

The original ONNX is an external, hash-pinned input. Default is plan-only;
--execute lowers unsupported pooling and --compile additionally invokes ATC.
This is a fixed [1,1682,80] deployment artifact, not a general speaker encoder.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from full_duplex_npu_speaker import DERIVED_GRAPH_SHA256, ORIGINAL_ONNX_SHA256


def lower_pool(node, shape, opset):
    import numpy as np
    from onnx import helper, numpy_helper

    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    expected = {
        "ceil_mode": 1,
        "kernel_shape": [100],
        "pads": [0, 0],
        "strides": [100],
    }
    if attributes != expected or len(shape) != 3 or not all(v > 0 for v in shape):
        raise ValueError("only the original disjoint ceil-mode pooling is supported")
    nodes, values, outputs = [], [], []
    for index, start in enumerate(range(0, shape[2], 100)):
        prefix = node.name + "/compat_window_" + str(index)
        names = []
        for suffix, data in [
            ("start", [start]),
            ("end", [min(start + 100, shape[2])]),
            ("axis", [2]),
        ]:
            name = prefix + "/" + suffix
            values.append(numpy_helper.from_array(np.array(data, dtype=np.int64), name))
            names.append(name)
        sliced, pooled = prefix + "/slice", prefix + "/mean"
        nodes.append(
            helper.make_node(
                "Slice", [node.input[0], *names], [sliced], name=prefix + "/Slice"
            )
        )
        if opset >= 18:
            mean = helper.make_node(
                "ReduceMean",
                [sliced, names[2]],
                [pooled],
                keepdims=1,
                name=prefix + "/ReduceMean",
            )
        else:
            mean = helper.make_node(
                "ReduceMean",
                [sliced],
                [pooled],
                axes=[2],
                keepdims=1,
                name=prefix + "/ReduceMean",
            )
        nodes.append(mean)
        outputs.append(pooled)
    nodes.append(
        helper.make_node(
            "Concat",
            outputs,
            list(node.output),
            axis=2,
            name=node.name + "/compat_Concat",
        )
    )
    return nodes, values


def prepare(original, output):
    # Check identity before parsing any graph or creating output.
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_ONNX_SHA256:
        raise ValueError("original published speaker ONNX hash differs")
    import onnx

    if onnx.__version__ != "1.19.1":
        raise ValueError("reproducible graph serialization requires onnx==1.19.1")
    model = onnx.load(original, load_external_data=False)
    if (
        len(model.graph.input) != 1
        or len(model.graph.input[0].type.tensor_type.shape.dim) != 3
    ):
        raise ValueError("one original rank-three input required")
    opset = next(o.version for o in model.opset_import if o.domain == "")
    for dim, value in zip(
        model.graph.input[0].type.tensor_type.shape.dim, [1, 1682, 80]
    ):
        dim.ClearField("dim_param")
        dim.dim_value = value
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in [*model.graph.input, *model.graph.value_info]
    }
    initializers = [t.SerializeToString() for t in model.graph.initializer]
    nodes, replaced = [], 0
    for node in model.graph.node:
        if node.op_type == "AveragePool":
            lowered, values = lower_pool(node, shapes[node.input[0]], opset)
            nodes.extend(lowered)
            model.graph.initializer.extend(values)
            replaced += 1
        else:
            nodes.append(node)
    if replaced != 52:
        raise ValueError("expected exactly 52 original pooling nodes")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if [
        t.SerializeToString() for t in model.graph.initializer[: len(initializers)]
    ] != initializers:
        raise ValueError("original neural parameters changed")
    onnx.checker.check_model(model)
    serialized = model.SerializeToString()
    if hashlib.sha256(serialized).hexdigest() != DERIVED_GRAPH_SHA256:
        raise ValueError("derived graph differs from the qualified deployment artifact")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "campplus.reference-shape.pool-lowered.onnx"
    destination.write_bytes(serialized)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--atc", default="/usr/local/Ascend/ascend-toolkit/latest/bin/atc"
    )
    args = parser.parse_args()
    if args.compile and not args.execute:
        parser.error("--compile requires --execute")
    original, output = args.original.resolve(), args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if original.is_relative_to(repository) or output.is_relative_to(repository):
        raise ValueError(
            "model input and generated artifacts must remain outside the repository"
        )
    report = {
        "status": "PLAN_ONLY",
        "original_sha256": ORIGINAL_ONNX_SHA256,
        "derived_sha256": DERIVED_GRAPH_SHA256,
        "input_shape": [1, 1682, 80],
        "neural_execution": False,
        "full_network_numeric_parity_verified": False,
        "npu_inference_qualified": False,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        print(json.dumps(report))
        return 0
    destination = prepare(original, output)
    report.update(
        status="GRAPH_PREPARED", original_initializers_unchanged=True, replaced_pools=52
    )
    code = 0
    if args.compile:
        command = [
            args.atc,
            "--framework=5",
            "--model=" + str(destination),
            "--output=" + str(output / "campplus"),
            "--input_format=ND",
            "--input_shape=input:1,1682,80",
            "--soc_version=Ascend910B2",
        ]
        # Persist a receipt even if compiler creation or compilation fails.
        try:
            with (output / "atc.log").open("xb") as log:
                result = subprocess.run(
                    command, cwd=output, stdout=log, stderr=subprocess.STDOUT
                )
            code = result.returncode
            report.update(
                compiler_exit_code=code,
                compiled_om_exists=(output / "campplus.om").is_file(),
            )
            if code == 0 and not report["compiled_om_exists"]:
                raise RuntimeError("ATC returned zero without a compiled model")
            report["status"] = (
                "COMPILED_NOT_INFERENCE_QUALIFIED"
                if code == 0
                else "COMPILATION_FAILED"
            )
        except Exception as error:
            report.update(
                status="COMPILATION_FAILED",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            (output / "qualification-public.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
    else:
        (output / "qualification-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(json.dumps(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
