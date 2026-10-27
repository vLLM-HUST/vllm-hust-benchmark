"""Check README navigation against delivery metadata; never claim native execution."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_readme_workload_catalog_covers_exact_delivery_scope():
    text = (ROOT / "README.md").read_text("utf-8")
    section = text.split("<!-- workload-catalog:start -->", 1)[1].split(
        "<!-- workload-catalog:end -->", 1
    )[0]
    classification = json.loads(
        (
            ROOT
            / "workload-preparation/data/catalogs/workload_delivery_classification_v1.json"
        ).read_text("utf-8")
    )
    inventory = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]
    ids = re.findall(r"^\|\s*(W\d{2})\s*\|", section, re.MULTILINE)
    assert ids == [row["workload_id"] for row in classification["classes"]]
    anchors = re.findall(r'<a id="workload-source-([^"]+)"></a>', section)
    assert len(anchors) == len(set(anchors)) == len(inventory) == 118
    assert set(anchors) == {name.lower().replace("_", "-") for name in inventory}
    for source, item in inventory.items():
        row = next(
            line
            for line in section.splitlines()
            if re.search(r"</a>" + re.escape(source) + r"\s*\|", line)
        )
        if item["checked"]:
            assert "[x] 已交付" in row
            assert item["delivery_receipt"] in row
        else:
            assert "[x]" not in row
        if source.startswith("PAIO-"):
            assert "原始数据私有" in row
        else:
            assert "[官方来源](https://" in row
    count = sum(item["checked"] for item in inventory.values())
    assert f"{count}/113 项已交付" in section
    assert "不等于全量模型评测" in section


def test_readme_catalog_local_links_exist_and_source_anchors_resolve():
    text = (ROOT / "README.md").read_text("utf-8")
    section = text.split("<!-- workload-catalog:start -->", 1)[1].split(
        "<!-- workload-catalog:end -->", 1
    )[0]
    for target in re.findall(r"\]\(([^)]+)\)", section):
        if target.startswith("https://"):
            continue
        if target.startswith("#workload-source-"):
            assert f'<a id="{target[1:]}"></a>' in section
        else:
            assert (ROOT / target.split("#", 1)[0]).is_file(), target
