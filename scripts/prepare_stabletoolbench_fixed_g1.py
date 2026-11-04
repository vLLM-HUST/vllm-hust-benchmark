"""Prepare fixed author G1 rows and tool definitions, without running the Agent."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


IDS = [588, 608, 1073]
ID_MAP_HASH = "b3dae86a67ce471413df4c734034b0c729357aebb233242a444cc2b8380f2f7f"
SUPPLEMENTS = ("kick_com_api_kick_api.json", "keyword_analysis.json")
HASHES = (
    "500aa0105ed451538448f6ea04f10661536b013f3e1a293840fe401688160e7c",
    "8f5b0bccbf5cf64b937c7d1364536c2adc568cf069c523838702c7e62eec8302",
    "a3ed72052c2dc961d8e9b71860374d90ded6d475ca6f6449c78fb248cb82a5b2",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("instructions", "toolenv", "cache", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument(
        "--official-ids",
        type=Path,
        help="Author test_query_ids/G1_instruction.json for FAC",
    )
    args = parser.parse_args()
    for path, expected in zip((args.instructions, args.toolenv, args.cache), HASHES):
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                parser.error(f"unexpected original asset bytes: {path}")
    rows = json.loads(args.instructions.read_bytes())[:3]
    if [row["query_id"] for row in rows] != IDS:
        parser.error("unexpected original row identities")
    fixed_ids = None
    if args.official_ids is not None:
        raw_ids = args.official_ids.read_bytes()
        if hashlib.sha256(raw_ids).hexdigest() != ID_MAP_HASH:
            parser.error("unexpected original FAC ID map bytes")
        official_ids = json.loads(raw_ids)
        fixed_ids = {str(task_id): official_ids[str(task_id)] for task_id in IDS}
    with tarfile.open(args.toolenv) as tools, zipfile.ZipFile(args.cache) as cache:
        members = tools.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not (member.isfile() or member.isdir())
                or path.is_absolute()
                or path.parts[:1] != ("toolenv2404_filtered",)
                or ".." in path.parts
                or "\\" in member.name
                or ":" in member.name
                or path in names
            ):
                parser.error(f"invalid tool archive member: {member.name}")
            names.add(path)
        definitions = {name: cache.read(f"tools/Data/{name}") for name in SUPPLEMENTS}
        args.output.mkdir(exist_ok=False)
        tools.extractall(args.output, members=members, filter="data")
        for name, raw in definitions.items():
            with (args.output / "toolenv2404_filtered" / "Data" / name).open(
                "xb"
            ) as stream:
                stream.write(raw)
        (args.output / "fixed-g1.json").write_text(json.dumps(rows), encoding="utf-8")
        if fixed_ids is not None:
            (args.output / "fixed-g1-fac-ids.json").write_text(
                json.dumps(fixed_ids), encoding="utf-8"
            )
    print(
        json.dumps({"prepared_ids": IDS, "model_calls": 0, "output": str(args.output)})
    )


if __name__ == "__main__":
    main()
