"""Prepare fixed NOMAD inputs from the competition authors' public release.

Uses the unchanged MLE-bench preparer and verifies its four published output
checksums. This is not the Kaggle download/outer ZIP verification workflow.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile


COMPETITION = "nomad2018-predict-transparent-conductors"
REVISION = "8ab1dd297a3692a3ec24fa9528c18bd0883ce85f"
SOURCE_URL = (
    f"https://codeload.github.com/csutton7/nomad_2018_kaggle_dataset/zip/{REVISION}"
)
SOURCE_BYTES = 6_957_213
SOURCE_SHA256 = "fbbc2667ff62b057d258725729f0506f48254f4c8606f44db01e587e405c89cf"
PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/mlebench-author-source.json"
)
GEOMETRY = re.compile(r"(train|test)/([0-9]+)/geometry\.xyz")
OUTPUTS = {
    "public": {"train.csv", "test.csv", "sample_submission.csv"},
    "private": {"test.csv"},
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_archive(archive):
    if archive.stat().st_size != SOURCE_BYTES or sha256(archive) != SOURCE_SHA256:
        raise ValueError(
            "public competition author archive does not match its fixed identity"
        )


def verify_installed_competition(package_root):
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    prefix = f"mlebench/competitions/{COMPETITION}/"
    files = [row for row in profile["files"] if row["path"].startswith(prefix)]
    if profile["source_id"] != "MLEBENCH" or len(files) != 8:
        raise ValueError("expected the eight pinned MLE-bench competition files")
    for row in files:
        path = package_root / row["path"].removeprefix("mlebench/")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(
                f"installed competition file does not match pin: {path.name}"
            )


def prepare(archive, output):
    validate_archive(archive)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("--data-dir must be a dedicated empty directory")

    import mlebench
    from mlebench.data import get_checksum, is_dataset_prepared
    from mlebench.registry import Registry
    from mlebench.utils import load_yaml

    verify_installed_competition(Path(mlebench.__file__).resolve().parent)
    competition = Registry(output).get_competition(COMPETITION)
    for directory in (
        competition.raw_dir,
        competition.public_dir,
        competition.private_dir,
    ):
        directory.mkdir(parents=True, mode=0o700)
    counts = {}
    prefix = f"nomad_2018_kaggle_dataset-{REVISION}/"
    with ZipFile(archive) as publisher:
        names = publisher.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate publisher archive paths")
        members = {n.removeprefix(prefix): n for n in names if n.startswith(prefix)}
        for split, count in (("train", 2400), ("test", 600)):
            name = f"{split}.csv"
            rows = publisher.read(members[name])
            reader = csv.DictReader(rows.decode("utf-8").splitlines())
            ids = [row["id"] for row in reader]
            files = sorted(
                n
                for n in members
                if GEOMETRY.fullmatch(n) and n.startswith(split + "/")
            )
            if len(ids) != len(set(ids)) or len(files) != count or len(ids) != count:
                raise ValueError(f"wrong row/geometry count for {split}")
            if {GEOMETRY.fullmatch(n)[2] for n in files} != set(ids):
                raise ValueError(f"wrong geometry identifiers for {split}")
            with ZipFile(
                competition.raw_dir / f"{split}.csv.zip", "x", ZIP_DEFLATED
            ) as target:
                target.writestr(name, rows)
            with ZipFile(
                competition.raw_dir / f"{split}.zip", "x", ZIP_DEFLATED
            ) as target:
                for file in files:
                    target.writestr(file, publisher.read(members[file]))
            counts[split] = count

    competition.prepare_fn(
        raw=competition.raw_dir,
        public=competition.public_dir,
        private=competition.private_dir,
    )
    if not is_dataset_prepared(competition):
        raise ValueError("the original preparer did not produce a complete dataset")
    expected = load_yaml(competition.checksums)
    verified = []
    for subset, directory in (
        ("public", competition.public_dir),
        ("private", competition.private_dir),
    ):
        if set(expected[subset]) != OUTPUTS[subset]:
            raise ValueError(f"unexpected published output checksum coverage: {subset}")
        for name, md5 in expected[subset].items():
            if get_checksum(directory / name) != md5:
                raise ValueError(
                    f"prepared file does not match published checksum: {subset}/{name}"
                )
            verified.append(f"{subset}/{name}")
    description = competition.public_dir / "description.md"
    with description.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(competition.description)
    return {
        "status": "AUTHOR_PUBLIC_DATA_PREPARED_AND_VERIFIED_NOT_AGENT_EXECUTION",
        "competition_id": COMPETITION,
        "author_archive_sha256": SOURCE_SHA256,
        "source_geometry_counts": counts,
        "prepared_checksums_verified": sorted(verified),
        "kaggle_outer_zip_verified": False,
        "publisher_labeled_test_extracted": False,
        "agent_mount_relative_to_data_dir": f"{COMPETITION}/prepared/public",
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.archive is None and args.data_dir is None:
        result = {
            "status": "PLAN_ONLY",
            "competition_id": COMPETITION,
            "archive_url": SOURCE_URL,
            "archive_bytes": SOURCE_BYTES,
            "archive_sha256": SOURCE_SHA256,
        }
    elif args.archive is None or args.data_dir is None:
        parser.error("--archive and --data-dir are both required for preparation")
    else:
        result = prepare(
            args.archive.resolve(strict=True), args.data_dir.resolve(strict=True)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
