"""Archive only the original long-probe bytes sealed by the midterm report."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"


def build():
    source = json.loads((REPORT / "results.json").read_text())
    runs = {row["run"] for row in source["long_context"]}
    members = []
    with tarfile.open(REPORT / "long-context-originals.tar.gz", "w:gz") as archive:
        for original, expected in sorted(source["source_sha256"].items()):
            path = Path(original)
            if (
                path.parent.name != "runtime-probe"
                or path.parent.parent.name not in runs
            ):
                continue
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == expected, original
            name = f"{path.parent.parent.name}/runtime-probe/{path.name}"
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(raw))
            members.append({"path": name, "sha256": expected, "bytes": len(raw)})
    assert len(members) == 172
    raw = (REPORT / "long-context-originals.tar.gz").read_bytes()
    (REPORT / "long-context-originals.json").write_text(
        json.dumps(
            {
                "schema": "midterm-long-probe-custody/v1",
                "scope": "Four original probe groups; no new experiment or latency inference.",
                "archive": "long-context-originals.tar.gz",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "members": members,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    build()
