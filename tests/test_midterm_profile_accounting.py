import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/recover_profile_accounting.py"
)
spec = importlib.util.spec_from_file_location("profile_accounting", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dense_transpose_is_respected():
    assert (
        module.matmul_flops(
            "MatMulV2", '"64,256;512,256"', '"64,512"', "transpose_x1:0|transpose_x2:1"
        )
        == 2 * 64 * 512 * 256
    )
    with pytest.raises(ValueError):
        module.matmul_flops(
            "MatMulV2", '"64,256;512,256"', '"64,512"', "transpose_x1:0|transpose_x2:0"
        )


def test_grouped_rows_are_not_multiplied_by_experts():
    assert (
        module.matmul_flops(
            "GroupedMatmul",
            '"12288,2048;256,2048,512;;;;;;256"',
            '"12288,512"',
            "transpose_weight:0|transpose_x:0|group_type:0|group_list_type:1",
        )
        == 2 * 12288 * 2048 * 512
    )


def test_unknown_layout_fails_closed():
    with pytest.raises(ValueError):
        module.matmul_flops("GroupedMatmul", '"8,4;16,4,2"', '"8,2"', "")
    with pytest.raises(ValueError):
        module.matmul_flops(
            "MatMulV3", '"1,8,4;1,4,2"', '"1,8,2"', "transpose_x1:0|transpose_x2:0"
        )


def test_graph_replays_count_tasks_not_static_definitions():
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE COMPUTE_TASK_INFO(globalTaskId PRIMARY KEY,opType,inputShapes,outputShapes,inputDataTypes,attrInfo)"
        )
        connection.execute("CREATE TABLE TASK(globalTaskId)")
        connection.execute("INSERT INTO COMPUTE_TASK_INFO VALUES(1,2,3,4,5,6)")
        connection.executemany("INSERT INTO TASK VALUES(?)", [(1,), (1,), (1,)])
        assert list(module.kernel_groups(connection)) == [(2, 3, 4, 5, 6, 3)]


def test_recovered_counts_match_independent_csv_inventory():
    recovered = json.loads(
        (PATH.parent / "m01-recovery/recovered-profile-accounting.json").read_text()
    )
    coverage = json.loads((PATH.parent / "aligned-profile-coverage.json").read_text())
    for rank, csv in zip(recovered["ranks"], coverage["files"], strict=True):
        for kind in ("MatMulV2", "MatMulV3", "GroupedMatmul"):
            assert rank["compute_task_counts"][kind] == csv["types"][kind]
        assert not rank["rejected_matmuls"]
        assert not rank["unknown_communication"]
        assert rank["whole_model_effective_flops"] is None
        assert rank["attainment_percent"] is None
        assert rank["matmul_shape_executed_flops"] == sum(
            module.matmul_flops(g["kind"], g["inputs"], g["outputs"], g["attrs"])
            * g["calls"]
            for g in rank["matmul_groups"]
        )
