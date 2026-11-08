"""Generate/check the original2700 Video-MME table; not a video/model run.

Offline plan by default. Execution uses only the pinned author's unchanged table
generator and verified original annotation file. Original media stay external.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"  # pragma: allowlist secret - fixed public dataset revision
SOURCE_SHA256 = "cc4ec6b863c7a5148fba791d87655ee6914809e1e079d29d2cf875039dc93b8e"  # pragma: allowlist secret - original source digest
ANNOTATION_SHA256 = "7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986"  # pragma: allowlist secret - original annotation digest
TABLE_MD5 = "85bdd91f9b29a99354c23b97ab7c113c"  # pragma: allowlist secret - original author table digest


def original_generator(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unverified original VideoMME implementation")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VideoMME"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_dataset"
    )
    functions = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_tsv"
    ]
    if len(functions) != 1:
        raise ValueError("Original table generator is ambiguous")
    return compile(
        ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
        "verified-original-videomme-generator",
        "exec",
    )


def prepare_view(checkout, snapshot):
    import os

    snapshot = Path(snapshot).resolve(strict=True)
    if snapshot.is_relative_to(ROOT) or ROOT.is_relative_to(snapshot):
        raise ValueError("External data root separate from repository required")
    if snapshot.name != REVISION:
        raise ValueError("Fixed original snapshot directory required")
    import pandas as pd

    code = original_generator(
        (Path(checkout) / "vlmeval/dataset/videomme.py").read_bytes()
    )
    annotation = snapshot / "videomme/test-00000-of-00001.parquet"
    if (
        annotation.stat().st_size != 405167
        or hashlib.sha256(annotation.read_bytes()).hexdigest() != ANNOTATION_SHA256
    ):
        raise ValueError("Original annotation file identity differs")
    raw = pd.read_parquet(annotation)
    if (
        len(raw) != 2700
        or raw["video_id"].nunique() != 900
        or raw["question_id"].nunique() != 2700
    ):
        raise ValueError("Complete original2700 question/900 video relations required")
    table = snapshot / "Video-MME.tsv"
    if table.exists() and hashlib.md5(table.read_bytes()).hexdigest() != TABLE_MD5:
        raise ValueError("Unexpected existing table retained; no overwrite")
    namespace = {
        "os": os,
        "osp": os.path,
        "pd": pd,
        "self": SimpleNamespace(MD5=TABLE_MD5),
        "dataset_name": "Video-MME",
        "md5": lambda path: hashlib.md5(Path(path).read_bytes()).hexdigest(),
    }
    exec(code, namespace)
    namespace["generate_tsv"](str(snapshot))
    if hashlib.md5(table.read_bytes()).hexdigest() != TABLE_MD5:
        raise ValueError("Generated table differs from original author identity")
    view = pd.read_csv(table, sep="\t")
    if len(view) != 2700:
        raise ValueError("Original derived-view row count differs")
    for index, row in view.iterrows():
        original = raw.iloc[index]
        if row["index"] != index or row["video"] != original["videoID"]:
            raise ValueError("Original video/index relation changed")
        if ast.literal_eval(row["candidates"]) != original["options"].tolist():
            raise ValueError("Original option order/content changed")
        if any(
            row[key] != original[key]
            for key in (
                "question",
                "answer",
                "duration",
                "domain",
                "sub_category",
                "task_type",
            )
        ):
            raise ValueError("Original task/reference/metadata changed")
        if (
            row["video_path"] != f"./video/{original['videoID']}.mp4"
            or row["subtitle_path"] != f"./subtitle/{original['videoID']}.srt"
        ):
            raise ValueError("Original video/subtitle locator relation changed")
    return {
        "source_id": "VIDEOMME",
        "status": "ORIGINAL2700_TABLE_AND_RELATIONS_VERIFIED_NOT_MEDIA_OR_NATIVE_EXECUTION",
        "original_tasks": 2700,
        "original_videos": 900,
        "author_table_md5": TABLE_MD5,
        "model_calls": 0,
        "source_checkbox": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        prepare_view(args.official_checkout, args.snapshot)
        if args.execute
        else {
            "source_id": "VIDEOMME",
            "status": "PLAN_ONLY",
            "original_tasks": 2700,
            "model_calls": 0,
            "downloads": False,
            "source_checkbox": False,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
