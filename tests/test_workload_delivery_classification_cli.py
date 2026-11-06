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


def test_original_execution_entries_are_discoverable_without_delivery_claim(capsys):
    expected = {
        "OSWORLD": "scripts/run_osworld_original.py",
        "MLAGENTBENCH": "scripts/run_mlagentbench_messages_agent.py",
        "NATUREBENCH": "scripts/run_naturebench_original.py",
    }
    for source, path in expected.items():
        main(["source-plan", source])
        payload = json.loads(capsys.readouterr().out)
        entries = {(row["role"], row["path"]) for row in payload["entrypoints"]}
        assert ("generation", path) in entries
        assert ("native_evaluation", path) in entries
