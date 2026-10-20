"""Recover logical operation counts from immutable profiler DBs."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3


DTYPE_BYTES = {
    "INT8": 1,
    "UINT8": 1,
    "INT16": 2,
    "UINT16": 2,
    "FP16": 2,
    "BFP16": 2,
    "INT32": 4,
    "UINT32": 4,
    "FP32": 4,
    "INT64": 8,
    "UINT64": 8,
    "FP64": 8,
    "INT128": 16,
}


def shapes(value):
    # The exported field is one quoted CSV cell containing semicolon-separated shapes.
    cell = next(csv.reader(io.StringIO(value)))
    if len(cell) != 1:
        raise ValueError("Expected one shape field")
    return [
        tuple(int(n) for n in part.split(",")) if part else ()
        for part in cell[0].split(";")
    ]


def matmul_flops(kind, inputs, outputs, attrs):
    inp, out = shapes(inputs), shapes(outputs)
    attributes = dict(part.split(":", 1) for part in attrs.split("|") if ":" in part)
    if len(out) != 1 or len(out[0]) != 2:
        raise ValueError("Unqualified output geometry")
    m, n = out[0]
    if kind in ("MatMulV2", "MatMulV3"):
        if len(inp) != 2 or any(len(s) != 2 for s in inp):
            raise ValueError("Only two-dimensional dense matmul is qualified")
        flags = [attributes.get("transpose_x1"), attributes.get("transpose_x2")]
        if any(flag not in ("0", "1") for flag in flags):
            raise ValueError("Missing transpose attributes")
        a, b = [s[::-1] if flag == "1" else s for s, flag in zip(inp, flags)]
        if a[0] != m or b[1] != n or a[1] != b[0]:
            raise ValueError("Matmul dimensions disagree")
        k = a[1]
    elif kind == "GroupedMatmul":
        required = {
            "transpose_weight": "0",
            "transpose_x": "0",
            "group_type": "0",
            "group_list_type": "1",
        }
        if any(attributes.get(key) != value for key, value in required.items()):
            raise ValueError("Unqualified grouped-matmul layout")
        if len(inp) < 2 or len(inp[0]) != 2 or len(inp[1]) != 3:
            raise ValueError("Unqualified grouped geometry")
        if inp[0][0] != m or inp[1][2] != n or inp[0][1] != inp[1][1]:
            raise ValueError("Grouped dimensions disagree")
        k = inp[0][1]
    else:
        raise ValueError("Unsupported operation")
    if min(m, n, k) <= 0:
        raise ValueError("Nonpositive dimension")
    return 2 * m * n * k


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def kernel_groups(connection):
    # Captured graphs reuse one definition across many TASK executions.
    return connection.execute(
        "SELECT c.opType,c.inputShapes,c.outputShapes,c.inputDataTypes,c.attrInfo,count(*) "
        "FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
        "GROUP BY c.opType,c.inputShapes,c.outputShapes,c.inputDataTypes,c.attrInfo"
    )


def recover(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    strings = dict(connection.execute("SELECT id,value FROM STRING_IDS"))
    dtypes = dict(connection.execute("SELECT id,name FROM ENUM_HCCL_DATA_TYPE"))
    rank_map = connection.execute(
        "SELECT rankId,deviceId FROM RANK_DEVICE_MAP"
    ).fetchall()
    communication = Counter()
    unknown = Counter()
    start, end = None, None
    for op_name, count, dtype, ranks, begin, finish in connection.execute(
        "SELECT opName,count,dataType,rankSize,startNs,endNs FROM COMMUNICATION_OP"
    ):
        name = strings[op_name]
        kind = (
            "all_reduce"
            if name.startswith("hcom_allReduce_")
            else "all_gather"
            if name.startswith("hcom_allGather_")
            else "unknown"
        )
        dtype_name = dtypes.get(dtype, str(dtype))
        if kind == "unknown" or dtype_name not in DTYPE_BYTES or ranks != 2:
            unknown[(kind, dtype_name, ranks)] += 1
            continue
        if not isinstance(count, int) or count < 0:
            raise ValueError("Invalid communication element count")
        communication[(kind, dtype_name, "calls")] += 1
        communication[(kind, dtype_name, "logical_input_bytes")] += (
            count * DTYPE_BYTES[dtype_name]
        )
        start = begin if start is None else min(start, begin)
        end = finish if end is None else max(end, finish)
    counts = Counter()
    missing = Counter()
    groups = []
    cursor = kernel_groups(connection)
    for kind_id, inp_id, out_id, dtype_id, attrs_id, repetitions in cursor:
        kind = strings.get(kind_id, str(kind_id))
        counts[kind] += repetitions
        if kind not in ("MatMulV2", "MatMulV3", "GroupedMatmul"):
            continue
        inp, out, attrs = (strings.get(i, "") for i in (inp_id, out_id, attrs_id))
        try:
            per_call = matmul_flops(kind, inp, out, attrs)
        except ValueError as error:
            missing[(kind, str(error))] += repetitions
            continue
        groups.append(
            dict(
                kind=kind,
                inputs=inp,
                outputs=out,
                dtype=strings.get(dtype_id),
                attrs=attrs,
                calls=repetitions,
                flops_per_call=per_call,
                shape_executed_flops=per_call * repetitions,
            )
        )
    connection.close()
    return dict(
        file=str(path),
        sha256=digest(path),
        rank_device_map=rank_map,
        communication_interval_ns=[start, end],
        communication=[
            dict(
                kind=kind,
                dtype=dtype,
                calls=communication[kind, dtype, "calls"],
                logical_input_bytes=communication[kind, dtype, "logical_input_bytes"],
            )
            for kind, dtype in sorted({(k, d) for k, d, _ in communication})
        ],
        unknown_communication=[
            dict(kind=k, dtype=d, ranks=r, calls=n)
            for (k, d, r), n in sorted(unknown.items())
        ],
        compute_task_counts=dict(counts),
        matmul_groups=groups,
        matmul_shape_executed_flops=sum(g["shape_executed_flops"] for g in groups),
        rejected_matmuls=[
            dict(kind=k, reason=r, calls=n) for (k, r), n in sorted(missing.items())
        ],
        physical_link_bytes=None,
        physical_hbm_bytes=None,
        whole_model_effective_flops=None,
        attainment_percent=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = sorted(
        args.trace_root.glob("*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db")
    )
    if len(files) != 2:
        raise ValueError("Expected both TP ranks")
    rows = [recover(path) for path in files]
    result = dict(
        schema="midterm-profiler-logical-accounting/v1",
        scope="Recovered diagnostic trace; not the unprofiled performance interval",
        matmul_definition="2*M*N*K per logical kernel shape, including any padding; grouped rows are total routed rows, not multiplied by expert count",
        communication_definition="Per-rank collective API input payload, count*dtype bytes; not sent+received or physical link traffic",
        exclusions="Non-matmul arithmetic, useful-vs-padded row distinction, physical HBM/link traffic and profiling overhead remain unqualified",
        ranks=rows,
        rank_communication_histograms_match=rows[0]["communication"]
        == rows[1]["communication"],
        rank_matmul_totals_match=rows[0]["matmul_shape_executed_flops"]
        == rows[1]["matmul_shape_executed_flops"],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "ranks"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
