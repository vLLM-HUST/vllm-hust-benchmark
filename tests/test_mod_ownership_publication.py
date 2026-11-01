"""Keep responsible people intact across catalog publication and metadata sync."""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "website-publication/data"


def test_responsible_people_are_preserved_in_both_publication_sources():
    catalog = json.loads((DATA / "ecosystem.json").read_text())
    metadata = json.loads((DATA / "plugin-workshop-metadata.json").read_text())
    components = {item["id"]: item for item in catalog["components"]}
    expected = {
        "muxpd": [("Irisuko", "毛潮云")],
        "kvcompress-ascend": [
            ("Jiawan23", "张家万"),
            ("kotoriqaq0", "韦若皓"),
            ("Seas0", "刘思辰"),
        ],
        "opfuse": [("KimmoZAG", "张睿诚")],
        "quality-bounded-inference": [("Tkhkrnx", "彭浩然")],
    }
    for key, people in expected.items():
        assert components[key]["maintainers"] == [login for login, _ in people]
        assert [
            (p["login"], p["name"]) for p in metadata["plugins"][key]["maintainers"]
        ] == people
    assert [a["name_zh"] for a in metadata["plugins"]["muxpd"]["advisors"]] == [
        "罗瑞坤"
    ]
    for key, item in metadata["plugins"].items():
        owners = [person["login"] for person in item["maintainers"]]
        assert owners == components[key]["maintainers"]
        assert all(login.lower() != "vllm-hust" for login in owners)
