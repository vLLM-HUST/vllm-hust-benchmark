"""Offline verification. --archive checks private evidence with the frozen client."""

import argparse
import hashlib
import json
import pathlib
import sys
import tarfile
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root, archive=None):
    root = pathlib.Path(root)
    if (root / "SHA256SUMS").exists():
        for line in (root / "SHA256SUMS").read_text().splitlines():
            h, name = line.split("  ", 1)
            p = pathlib.Path(name)
            assert not p.is_absolute() and ".." not in p.parts
            assert sha((root / p).read_bytes()) == h
    proof = json.loads((root / "checkpoint.json").read_text())
    assert proof["state"] == "NATIVE_QUALITY_COMPLETE_CONTROLLER_PENDING"
    report = proof["summary"]
    assert (
        report["scope"] == "full_gold"
        and report["denominator_policy"] == "all_cases_invalid_as_nonsuccess"
    )
    for track, n in [("olympiad", 100), ("research", 60)]:
        t = report["tracks"][track]
        assert (
            t["total"] == n
            and t["graded"] + t["judge_unresolved"] + t["solver_failures"] == n
        )
        assert t["success_rate"] == t["successes"] / n
    if archive:
        assert sha(pathlib.Path(archive).read_bytes()) == proof["private_raw_sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    p = pathlib.PurePosixPath(member.name)
                    assert (
                        not p.is_absolute()
                        and ".." not in p.parts
                        and (member.isdir() or member.isfile())
                    )
                    dest = base.joinpath(*p.parts)
                    if member.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(tar.extractfile(member).read())
            plan = base / "plan"
            assert (
                sha((plan / "contract.json").read_bytes()) == proof["contract_sha256"]
            )
            assert sha((plan / "tasks.json").read_bytes()) == proof["tasks_sha256"]
            package = base / "hust_eval_frontierscience"
            package.mkdir()
            for name, h in proof["client_code_sha256"].items():
                assert pathlib.Path(name).name == name
                data = (plan / "client_source" / name).read_bytes()
                assert sha(data) == h
                (package / name).write_bytes(data)
            sys.path.insert(0, str(base))
            from hust_eval_frontierscience.pipeline import (
                load_plan,
                verify_run,
                digest,
                request_body,
                parse_grade,
                summary,
            )

            c, tasks = load_plan(plan)
            assert c["client_code"] == proof["client_code_sha256"]
            ch = proof["contract_sha256"]
            sm, sr = verify_run(base / "solver", ch, "solver")
            jm, jr = verify_run(base / "judge", ch, "judge")
            assert len(sr) == len(jr) == 160
            assert set(sr) == set(jr) == {t["case_id"] for t in tasks}
            for task in tasks:
                s = sr[task["case_id"]]
                j = jr[task["case_id"]]
                messages = {
                    "solver": [{"role": "user", "content": task["problem"]}],
                    "judge": [
                        {
                            "role": "user",
                            "content": c["judge_prompts"][task["track"]].format_map(
                                {
                                    "problem": task["problem"],
                                    "answer": s.get("content") or "",
                                    "reference answer": task["answer"],
                                    "rubric": task["answer"],
                                }
                            ),
                        }
                    ],
                }
                for kind, r in [("solver", s), ("judge", j)]:
                    copy = dict(r)
                    h = copy.pop("record_sha256")
                    assert digest(copy) == h
                    assert r["request"] == request_body(c[kind], messages[kind])
                    assert digest(r["request"]) == r["request_sha256"]
                    assert r["contract_sha256"] == ch and r["track"] == task["track"]
                    if r["status"] in ("ok", "invalid_judge"):
                        response = json.loads(r["raw_response"])
                        assert response == r["response"]
                        choice = response["choices"][0]
                        assert (
                            choice["message"]["content"] == r["content"]
                            and choice["finish_reason"] == r["finish_reason"]
                        )
                        assert response["usage"] == r["usage"]
                        u = r["usage"]
                        assert all(
                            type(u[k]) is int and u[k] >= 0
                            for k in [
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                            ]
                        )
                        assert (
                            u["prompt_tokens"] + u["completion_tokens"]
                            == u["total_tokens"]
                        )
                assert j["solver_record_sha256"] == digest(s)
                if j["status"] == "ok":
                    assert (
                        parse_grade(
                            task["track"],
                            j["content"],
                            j["finish_reason"],
                            c["scoring_policy"],
                        )
                        == j["grade"]
                    )
                elif j["status"] == "invalid_judge":
                    try:
                        parse_grade(
                            task["track"],
                            j["content"],
                            j["finish_reason"],
                            c["scoring_policy"],
                        )
                    except (ValueError, TypeError) as e:
                        assert str(e) == j["error"]
                    else:
                        raise AssertionError("Invalid judge is parseable")
            out = base / "recomputed"
            summary(
                argparse.Namespace(
                    plan=plan,
                    responses=base / "solver",
                    grades=base / "judge",
                    output=out,
                )
            )
            for name in ["summary.json", "per_case.csv"]:
                assert (out / name).read_bytes() == (
                    base / "report" / name
                ).read_bytes()
            assert json.loads((out / "summary.json").read_text()) == report
    return {
        "checkpoint": "PASS",
        "private_archive_verified": bool(archive),
        "native_quality": {
            t: {
                k: report["tracks"][t][k]
                for k in [
                    "total",
                    "successes",
                    "success_rate",
                    "judge_unresolved",
                    "solver_failures",
                ]
            }
            for t in ["olympiad", "research"]
        },
        "controller_comparison": "PENDING",
        "note": "Public-only mode checks proof consistency, not private responses or answer correctness.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive")
    a = p.parse_args()
    print(json.dumps(verify(pathlib.Path(__file__).parent, a.archive)))
