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
REVISION = "8ab1dd297a3692a3ec24fa9528c18bd0883ce85f"  # pragma: allowlist secret - public Git revision
SOURCE_URL = (
    f"https://codeload.github.com/csutton7/nomad_2018_kaggle_dataset/zip/{REVISION}"
)
SOURCE_BYTES = 6_957_213
SOURCE_SHA256 = "fbbc2667ff62b057d258725729f0506f48254f4c8606f44db01e587e405c89cf"  # pragma: allowlist secret - public archive checksum
PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/mlebench-author-source.json"
)
GEOMETRY = re.compile(r"(train|test)/([0-9]+)/geometry\.xyz")
OUTPUTS = {
    "public": {"train.csv", "test.csv", "sample_submission.csv"},
    "private": {"test.csv"},
}
PREPARED_COUNTS = {"train": 2160, "test": 240}


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


def verify_public_structure(public):
    expected = OUTPUTS["public"] | {"description.md", "train", "test"}
    if {path.name for path in public.iterdir()} != expected or any(
        path.is_symlink() for path in public.rglob("*")
    ):
        raise ValueError("unexpected paths in the prepared public directory")

    def read_table(name):
        with (public / name).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, [row["id"] for row in reader]

    train_fields, train_ids = read_table("train.csv")
    test_fields, test_ids = read_table("test.csv")
    sample_fields, sample_ids = read_table("sample_submission.csv")
    targets = set(sample_fields) - {"id"}
    if sample_ids != test_ids or not targets or not targets <= set(train_fields):
        raise ValueError("sample submission does not match the original public tables")
    if targets & set(test_fields):
        raise ValueError("public test table unexpectedly contains target columns")

    for split, ids in (("train", train_ids), ("test", test_ids)):
        root = public / split
        if len(ids) != PREPARED_COUNTS[split] or len(set(ids)) != len(ids):
            raise ValueError(f"unexpected prepared row count for {split}")
        folders = {path.name: path for path in root.iterdir()}
        if set(folders) != set(ids) or any(
            not folder.is_dir() for folder in folders.values()
        ):
            raise ValueError(f"geometry folders do not match prepared IDs for {split}")
        for folder in folders.values():
            geometry = folder / "geometry.xyz"
            if not geometry.is_file() or not geometry.stat().st_size:
                raise ValueError(f"missing or empty original geometry in {split}")
            if {path.name for path in folder.iterdir()} != {"geometry.xyz"}:
                raise ValueError(f"unexpected file beside original geometry in {split}")

    return {
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "sample_submission_rows": len(sample_ids),
        "geometry_references_resolved": len(train_ids) + len(test_ids),
        "test_and_sample_submission_order_matched": True,
        "test_target_columns_present": False,
    }


def verify_public(public):
    import mlebench
    from mlebench.data import get_checksum
    from mlebench.utils import load_yaml

    package = Path(mlebench.__file__).resolve().parent
    verify_installed_competition(package)
    expected = load_yaml(package / "competitions" / COMPETITION / "checksums.yaml")[
        "public"
    ]
    if set(expected) != OUTPUTS["public"]:
        raise ValueError("unexpected published public output checksum coverage")
    readback = verify_public_structure(public)
    for name, checksum in expected.items():
        if get_checksum(public / name) != checksum:
            raise ValueError(f"prepared public file does not match publisher: {name}")
    return {
        "status": "PREPARED_PUBLIC_INPUT_VERIFIED_NOT_AGENT_EXECUTION",
        "competition_id": COMPETITION,
        **readback,
        "publisher_public_table_checksums_verified": sorted(expected),
        "private_scoring_labels_read": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--verify-public-dir", type=Path)
    args = parser.parse_args()
    if args.verify_public_dir is not None:
        if args.archive is not None or args.data_dir is not None:
            parser.error("--verify-public-dir cannot be combined with preparation")
        result = verify_public(args.verify_public_dir.resolve(strict=True))
    elif args.archive is None and args.data_dir is None:
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
