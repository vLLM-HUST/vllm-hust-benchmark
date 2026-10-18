import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    h, name = line.split("  ", 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == h, name
p = json.loads((root / "publication.json").read_text())
assert p["status"] == "ONLINE_UI_VERIFIED"
assert p["pages"]["status"] == "built" and p["pages"]["commit"] == p["expected_merge"]
assert p["pr"] == 441
assert {(v["width"], v["language"]) for v in p["viewports"]} == {
    (1440, "zh"),
    (390, "en"),
    (320, "zh"),
}
for v in p["viewports"]:
    assert (
        v["status"] == "PASS"
        and v["quality_rows"] == 5
        and v["replay_rows"] == 1
        and v["records"] == 8
        and not v["errors"]
    )
print("PASS published presentation receipt; not an additional model experiment")
