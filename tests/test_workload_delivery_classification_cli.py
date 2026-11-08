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


def test_replacement_execution_entries_identify_the_actual_source(capsys):
    replacements = {
        "TOOLBENCH": ("STABLETOOLBENCH", "scripts/run_stabletoolbench_local_server.py"),
        "VISUALWEBARENA": ("MINIWOB_PLUSPLUS", "scripts/run_miniwob_agentlab.py"),
        "WORKARENA": ("WORKBENCH", "scripts/run_workbench_calendar_native.py"),
        "PAPERBENCH": (
            "RESEARCHCODEBENCH",
            "scripts/run_researchcodebench_fixed3_native.py",
        ),
    }
    for slot, (replacement, path) in replacements.items():
        main(["source-plan", slot])
        payload = json.loads(capsys.readouterr().out)
        (entry,) = [
            row for row in payload["entrypoints"] if row.get("execution_source_id")
        ]
        assert entry["execution_source_id"] == replacement
        assert entry["path"] == path


def test_bbh_source_plan_includes_original_archive_preparation(capsys):
    main(["source-plan", "BBH"])
    payload = json.loads(capsys.readouterr().out)
    assert any(
        entry["role"] == "prepare_assets"
        and entry["path"] == "src/vllm_hust_benchmark/workload_data.py"
        and "workload-data prepare-bbh-original" in entry["scope"]
        for entry in payload["entrypoints"]
    )
