"""Metadata CLI integration, never source acquisition or model execution."""

import json

from vllm_hust_benchmark.workload_data import main


def test_class_and_asset_views(capsys):
    main(["classes", "--scheme", "delivery-plan"])
    assert len(json.loads(capsys.readouterr().out)) == 40
    main(["delivery-assets"])
    assert len(json.loads(capsys.readouterr().out)) == 118
    main(["source-plan", "PAIO-JSON-500"])
    assert json.loads(capsys.readouterr().out)["related_workload_ids"] == ["W06"]
