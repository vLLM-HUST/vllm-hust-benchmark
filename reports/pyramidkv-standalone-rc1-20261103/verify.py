"""Verify the standalone candidate's raw functional acceptance, offline."""

import hashlib
import json
import re
import tarfile
from pathlib import Path


def verify(root):
    if not __debug__:
        raise RuntimeError(
            "Verification requires assertions; omit -O and PYTHONOPTIMIZE"
        )
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = root / name
        assert path.parent == root and path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    expected = json.loads((root / "members.json").read_text())
    with tarfile.open(root / "raw.tar.gz") as archive:
        entries = archive.getmembers()
        assert all(item.isfile() for item in entries)
        assert len(entries) == len({item.name for item in entries})
        raw = {item.name: archive.extractfile(item).read() for item in entries}
    assert set(raw) == set(expected)
    for name, content in raw.items():
        assert len(content) == expected[name]["bytes"], name
        assert hashlib.sha256(content).hexdigest() == expected[name]["sha256"], name

    def read(name):
        return json.loads(raw[name])

    def total(text, name):
        values = re.findall(
            rf"^{re.escape(name)}(?:\{{[^\n]*\}})?\s+([\d.eE+-]+)$",
            text,
            re.MULTILINE,
        )
        assert values, name
        return sum(map(float, values))

    counts = {}
    prompts = {}
    for arm in ("disabled", "native", "batched"):
        prefix = f"acceptance/{arm}/"
        assert read(prefix + "summary.json")["passed"] is True
        assert "failure.json" not in {
            n.removeprefix(prefix) for n in raw if n.startswith(prefix)
        }
        log = raw[prefix + "server.log"].decode(errors="replace")
        launch = raw[prefix + "serve.sh"].decode()
        assert f"PYRAMIDKV_ENABLED={'0' if arm == 'disabled' else '1'}" in launch
        assert (
            f"VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA={'0' if arm == 'native' else '1'}"
            in launch
        )
        assert "0.4.0rc1" in log
        if arm == "disabled":
            assert "PyramidKV plugin activated" not in log
            assert "scheduler commit request_id=" not in log
            assert "worker commit acknowledged request_id=" not in log
        else:
            assert "PyramidKV plugin activated" in log
        records = []
        transactions = []
        for name, answer in [
            ("short", "2"),
            ("long", "长文本测试成功。"),
            ("long-repeat", "长文本测试成功。"),
            *[(f"concurrent-{i}", "7") for i in range(4)],
        ]:
            request = read(prefix + name + "-request.json")
            response = read(prefix + name + "-response.json")
            assert response["choices"][0]["message"]["content"].strip() == answer
            if name in prompts:
                assert request == prompts[name], (arm, name)
            else:
                prompts[name] = request
            records.append(response)
            if name != "short":
                assert response["usage"]["prompt_tokens"] > 4096
                if arm != "disabled":
                    identity = re.escape(response["id"])
                    matches = re.findall(
                        rf"scheduler commit request_id=({identity}(?:-[a-zA-Z0-9]+)?) "
                        rf"semantic_tokens=(\d+) physical_tokens=(\d+)",
                        log,
                    )
                    assert len(matches) == 1, (arm, name)
                    engine, semantic, physical = matches[0]
                    assert int(semantic) == response["usage"]["prompt_tokens"]
                    assert 0 < int(physical) < int(semantic)
                    for rank in (0, 1):
                        acknowledgements = re.findall(
                            rf"Worker_TP{rank} [^\n]*worker commit acknowledged "
                            rf"request_id={re.escape(engine)} "
                            rf"semantic_tokens=(\d+) physical_tokens=(\d+)",
                            log,
                        )
                        assert acknowledgements == [(semantic, physical)]
                    transactions.append(engine)
        if arm != "disabled":
            assert len(re.findall("scheduler commit request_id=", log)) == 6
            assert len(set(transactions)) == 6
        assert records[2]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        before = raw[prefix + "metrics-before.txt"].decode()
        after = raw[prefix + "metrics-after.txt"].decode()
        for metric in ("num_draft_tokens", "num_accepted_tokens"):
            name = "vllm:spec_decode_" + metric + "_total"
            assert total(after, name) - total(before, name) > 0
        counts[arm] = {"requests": len(records), "transactions": len(transactions)}
    identities = [read(f"acceptance/{arm}/process.json")["pid"] for arm in counts]
    assert len(set(identities)) == 3
    print(json.dumps({"verified": True, "members": len(raw), "arms": counts}))


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
