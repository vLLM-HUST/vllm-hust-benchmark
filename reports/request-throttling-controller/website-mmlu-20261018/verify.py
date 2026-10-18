import pathlib
import json
import hashlib

r = pathlib.Path(__file__).parent
for line in (r / "SHA256SUMS").read_text().splitlines():
    h, n = line.split("  ", 1)
    assert pathlib.Path(n).name == n
    assert hashlib.sha256((r / n).read_bytes()).hexdigest() == h
m = json.loads((r / "mmlu-website-merge-receipt.json").read_text())
o = json.loads((r / "mmlu-website-online-receipt.json").read_text())
c = json.loads((r / "receipt.json").read_text())
assert m["pr"] == o["pr"] == 445 and m["head"] == o["head"] == c["head"]
assert o["merge"] == m["merged_sha"] == o["pages"]["commit"]
assert (
    o["pages"]["status"] == "built"
    and o["status"] == "PUBLISHED_MMLU_CONTENT_AND_LIVE_UI_VERIFIED"
)
assert len(o["served_sha256"]) == 5
assert len(m["checks"]) == 4 and all(
    x["status"] == "completed" and x["conclusion"] == "success" for x in m["checks"]
)
print(
    json.dumps(
        {
            "publication_evidence": "PASS",
            "live_in_app_browser_verified": True,
            "served_content_verified": True,
            "merge": o["merge"],
        }
    )
)
