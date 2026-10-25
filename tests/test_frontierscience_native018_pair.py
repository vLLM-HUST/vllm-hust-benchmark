import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/frontierscience-native018-paired-20261025"


def test_packet_hashes_and_full_denominators():
    runtime_raw = (DATA / "b0-runtime-identity.json").read_bytes()
    assert (
        hashlib.sha256(runtime_raw).hexdigest()
        == "1bce4f4e86ff2a8fc0cef361bffd2e01872d1d58d4ddffacaf7fc1441b2ed2d9"  # pragma: allowlist secret
    )
    runtime = json.loads(runtime_raw)
    assert runtime["vllm_version"] == runtime["ascend_version"] == "0.18.0"
    assert runtime["argv"][runtime["argv"].index("--max-model-len") + 1] == "32768"
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        assert (
            hashlib.sha256((DATA / name.removeprefix("./")).read_bytes()).hexdigest()
            == sha
        )
    for prefix, expected in (("b0", [66, 3]), ("paired-b1", [62, 2])):
        summary = json.loads((DATA / f"{prefix}-summary.json").read_text())
        with (DATA / f"{prefix}-per-case.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len({r["case_id"] for r in rows}) == 160
        for track, total, successes in zip(
            ("olympiad", "research"), (100, 60), expected
        ):
            selected = [r for r in rows if r["track"] == track]
            observed = summary["tracks"][track]
            assert len(selected) == observed["total"] == total
            assert (
                observed["successes"]
                == successes
                == sum(
                    r["judge_status"] == "ok" and r["correct"] == "True"
                    for r in selected
                )
            )
            assert observed["judge_unresolved"] == sum(
                r["judge_status"] != "ok" for r in selected
            )
            for key in ("solver_truncated", "judge_truncated"):
                assert observed[key] == sum(r[key] == "True" for r in selected)


def test_v082_preserves_m01_and_reports_negative_quality(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v082 as builder

    data = builder.build(
        "test",
        lambda p: (
            ROOT / (p.replace("中期测试报告.md", "中期测试报告-v082.md"))
        ).read_bytes(),
    )
    old = json.loads((ROOT / "website-publication/data/midterm-v081.json").read_text())
    for key in (
        "wholechain_reference",
        "paired_own_compute",
        "m01_method",
        "corrected_quality",
    ):
        assert data[key] == old[key]
    pair = data["frontierscience_designated_pair"]
    assert pair["quality_threshold"] is None and pair["quality_passed"] is None
    for track in ("olympiad", "research"):
        assert pair["B1"][track]["successes"] < pair["B0"][track]["successes"]
    section = next(
        s for s in data["sections"] if s["id"] == "frontierscience-designated-pair"
    )
    assert [r[-1] for r in section["tables"][0]["rows"]] == ["-4.0000", "-1.6667"]
