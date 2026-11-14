"""Synthetic provenance tests; no model startup or performance claims."""

import copy
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("publication_campaign", Path(__file__).resolve().parents[1] / "scripts/publication_campaign.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def campaign(reused):
    rows = []
    sources = {}
    for i, name in enumerate(MODULE.STAGES):
        row = dict(name=name, status="completed", returncode=0, started=100 + i, ended=200 + i)
        if i < reused:
            row.update(status="reused_verified", source=dict(source_index=i, source_stage_receipt=dict(started=i, ended=i + 1)))
            sources[name] = f"/old/{name}"
        rows.append(row)
    return dict(receipt=dict(status="completed", selected_execution_complete=True, stages=rows,
                             reused_inference_stages=reused, new_inference_stages=15 - reused, ended=1794672000),
                reused_stage_sources=sources,
                plan=dict(datasets=["mmlu", "hle", "frontier", "swe", "terminal"],
                          smoke_evaluators=False, hle_profile="fixed200-v1", mmlu_inputs=dict(requests=200)))


def test_mixed_campaign_retains_each_actual_source_and_original_time():
    data = campaign(2)
    original = copy.deepcopy(data)
    assert MODULE.validate(data) == (2, 13)
    roots = MODULE.roots(Path("/new"), data)
    assert roots["continuous-B0"] == Path("/old/continuous-B0")
    assert roots["mmlu-pair"] == Path("/new/mmlu-pair")
    rows = MODULE.provenance(data)
    assert rows[0]["original_started"] == 0
    assert rows[2]["original_started"] == 102
    assert rows[2]["source_index"] is None
    assert data == original


def test_all_new_and_all_reused_are_both_supported():
    for n in (0, 15):
        data = campaign(n)
        assert MODULE.validate(data) == (n, 15 - n)
        assert len(MODULE.roots(Path("/new"), data)) == 15


def test_incomplete_failed_missing_and_inconsistent_campaigns_are_rejected():
    variants = []
    data = campaign(2); data["receipt"]["status"] = "failed"; variants.append(data)
    data = campaign(2); data["receipt"]["stages"][-1]["status"] = "planned"; variants.append(data)
    data = campaign(2); data["receipt"]["stages"].pop(); variants.append(data)
    data = campaign(2); data["receipt"]["reused_inference_stages"] = 15; variants.append(data)
    data = campaign(2); data["reused_stage_sources"].pop("continuous-B0"); variants.append(data)
    data = campaign(2); data["plan"]["smoke_evaluators"] = True; variants.append(data)
    data = campaign(2); data["plan"]["hle_profile"] = "full1811"; variants.append(data)
    for data in variants:
        try:
            MODULE.validate(data)
        except ValueError:
            continue
        raise AssertionError("Unsafe synthetic campaign was accepted")


def test_device_identity_follows_each_stage_including_reused_origins(tmp_path):
    for name, cards in (("reused", [5, 6]), ("new", [1, 6])):
        root = tmp_path / name
        root.mkdir()
        (root / "device-mapping.json").write_text(json.dumps(dict(physical_cards=cards)))
        assert MODULE.device_cards(root) == tuple(map(str, cards))


def test_missing_duplicate_or_invalid_device_identity_cannot_use_legacy_defaults(tmp_path):
    for cards in ([1, 1], [True, 6], ["1", "6"], [1], [-1, 6]):
        (tmp_path / "device-mapping.json").write_text(json.dumps(dict(physical_cards=cards)))
        try:
            MODULE.device_cards(tmp_path)
        except ValueError:
            continue
        raise AssertionError("Invalid physical device identity accepted")
    (tmp_path / "device-mapping.json").unlink()
    try:
        MODULE.device_cards(tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("Missing physical device identity accepted")
