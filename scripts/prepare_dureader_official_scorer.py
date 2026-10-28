"""Prepare original DuReader scoring software, never dataset/model validation.

The publisher downloader uses coco-caption master and 2to3 for BLEU. This
entrypoint pins both repositories and records that same explicit conversion.
The evaluation module additionally needs Python2 dict snapshot semantics when
deleting empty references during iteration; only standard fix_dict is applied.
No prediction, raw dataset or model is downloaded. Use a fresh private output.
"""

import argparse
import base64
import hashlib
import importlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

DUREADER_REVISION = "c625076b06da8f56d59f19c41c73bd580a98a347"  # pragma: allowlist secret - public publisher commit
COCO_REVISION = "3a9afb2682141a03e1cdc02b0df6770d2c884f6f"  # pragma: allowlist secret - public metric dependency commit
FILES = {
    "dureader_official/__init__.py": (
        "baidu/DuReader",
        DUREADER_REVISION,
        "DuReader-2.0/utils/__init__.py",
    ),
    "dureader_official/dureader_eval.py": (
        "baidu/DuReader",
        DUREADER_REVISION,
        "DuReader-2.0/utils/dureader_eval.py",
    ),
    "dureader_official/preprocess.py": (
        "baidu/DuReader",
        DUREADER_REVISION,
        "DuReader-2.0/utils/preprocess.py",
    ),
    "publisher-download_thirdparty.sh": (
        "baidu/DuReader",
        DUREADER_REVISION,
        "DuReader-2.0/utils/download_thirdparty.sh",
    ),
}
for metric, names in {
    "bleu": ("LICENSE", "__init__.py", "bleu.py", "bleu_scorer.py"),
    "rouge": ("__init__.py", "rouge.py"),
}.items():
    for name in names:
        FILES[f"dureader_official/{metric}_metric/{name}"] = (
            "tylin/coco-caption",
            COCO_REVISION,
            f"pycocoevalcap/{metric}/{name}",
        )


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def github_fetch(url):
    """Publisher GitHub API route, not a third-party code mirror."""
    prefix = "https://raw.githubusercontent.com/"
    if not url.startswith(prefix):
        raise ValueError("original GitHub software URL required")
    owner, repository, revision, path = url[len(prefix) :].split("/", 3)
    request = Request(
        f"https://api.github.com/repos/{owner}/{repository}/contents/{path}?ref={revision}",
        headers={
            "User-Agent": "vllm-hust-native-delivery",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=90) as response:
        info = json.loads(response.read(2 * 1024**2 + 1))
    if (
        info.get("type") != "file"
        or info.get("path") != path
        or info.get("encoding") != "base64"
    ):
        raise ValueError("publisher API software metadata mismatch")
    payload = base64.b64decode(info["content"])
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if len(payload) != info.get("size") or blob != info.get("sha"):
        raise ValueError("publisher API software bytes mismatch")
    return payload


def prepare(output, *, fetch=None):
    """Download software with TLS verification, fail rather than replace state."""
    requested = Path(output).absolute()
    if requested.exists() or requested.is_symlink():
        raise ValueError("fresh output required; existing state is never overwritten")
    output = requested.resolve()
    output.mkdir(parents=True)
    if fetch is None:

        def fetch(url):
            request = Request(url, headers={"User-Agent": "vllm-hust-native-delivery"})
            with urlopen(request, timeout=90) as response:
                return response.read()

    # Import only on the supported Python3.12 preparation runtime.
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package

    converter = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    records, originals = [], {}
    for relative, (repository, revision, publisher_path) in FILES.items():
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/{publisher_path}"
        original = fetch(url)
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original)
        originals[relative] = (original, url)
    # As in the author's download script, obtain every module before 2to3.
    # fix_import uses sibling module existence to resolve relative imports.
    for relative, (original, url) in originals.items():
        path = output / relative
        converted = original
        if relative.endswith(("/bleu.py", "/bleu_scorer.py")):
            converted = str(
                converter.refactor_string(original.decode("utf-8"), str(path))
            ).encode("utf-8")
        elif relative.endswith("/dureader_eval.py"):
            snapshot = RefactoringTool(["lib2to3.fixes.fix_dict"])
            converted = str(
                snapshot.refactor_string(original.decode("utf-8"), str(path))
            ).encode("utf-8")
        path.write_bytes(converted)
        records.append(
            {
                "path": relative,
                "publisher_url": url,
                "original_sha256": digest(original),
                "installed_sha256": digest(converted),
                "conversion": (
                    "Python2 dictionary snapshot compatibility via standard 2to3 fix_dict"
                    if relative.endswith("/dureader_eval.py")
                    else "publisher-recommended standard 2to3 compatibility"
                )
                if converted != original
                else "none",
            }
        )
    return {
        "status": "SCORING_SOFTWARE_PREPARED_NOT_NATIVE_DATASET_VALIDATION",
        "files": records,
        "model_calls": 0,
    }


def fixture_check(output):
    """Synthetic annotation fixtures only; no native task success claim."""
    sys.path.insert(0, str(Path(output).resolve()))
    module = importlib.import_module("dureader_official.dureader_eval")
    rows = {
        1: {
            "question_type": "DESCRIPTION",
            "answers": ["fixture description"],
            "yesno_answers": [],
            "entity_answers": [[]],
            "source": "search",
        },
        2: {
            "question_type": "ENTITY",
            "answers": ["fixture entity"],
            "yesno_answers": [],
            "entity_answers": [["fixture entity"]],
            "source": "search",
        },
        3: {
            "question_type": "YES_NO",
            "answers": ["fixture yes", "fixture no", "fixture depends"],
            "yesno_answers": ["Yes", "No", "Depends"],
            "entity_answers": [[]],
            "source": "search",
        },
    }
    # Exactly one prediction per question; all original-style reference labels
    # remain on the reference side. Empty label channels are handled upstream.
    predictions = {
        key: {
            **row,
            "answers": row["answers"][:1],
            "yesno_answers": row["yesno_answers"][:1],
        }
        for key, row in rows.items()
    }
    scores = {}
    for task in ("main", "all", "description", "entity", "yesno"):
        try:
            scores[task] = module.get_metrics(predictions, rows, task, "both")
        except (AssertionError, RuntimeError) as error:
            # Retain compatibility failures, never silently alter metric code.
            scores[task] = {
                "software_compatibility_error": type(error).__name__,
                "detail": str(error),
            }
    if scores["entity"]["F1"] != 1.0 or scores["description"]["Rouge-L"] < 0.99:
        raise ValueError("publisher scoring software fixture failed")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--github-api",
        action="store_true",
        help="Explicit original publisher GitHub contents API transport",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise ValueError("pinned preparation profile requires Python3.12")
    receipt = prepare(args.output, fetch=github_fetch if args.github_api else None)
    receipt["transport"] = (
        "publisher GitHub API" if args.github_api else "publisher raw GitHub"
    )
    receipt["synthetic_software_fixture_scores"] = fixture_check(args.output)
    receipt["python"] = sys.version
    (args.output / "receipt-public.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
