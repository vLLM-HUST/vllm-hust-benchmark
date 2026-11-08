"""Fixed original public-test DST preparation, not speech or model execution.

Original publisher source stays external. The compatibility copy only flushes
two intermediate files before reread; task/prompt/reference logic is unchanged.
"""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

REPOSITORY = "AlibabaResearch/DAMO-ConvAI"
REVISION = "faf754671f5b850e1b4d1ff870db37a6a2480df6"  # pragma: allowlist secret - public author revision
PINS = {
    "dst_data.py": "03eab2c776b094faf5cae7a6d501b9bf478a6827",  # pragma: allowlist secret
    "eval_JGA.py": "d5242cf0af8d800ad05099e0fb56fbe97e2e0879",  # pragma: allowlist secret
    "request_openai.py": "e8bbc1e2da476d7c62b08420755f04853eb63ced",  # pragma: allowlist secret
}
IDS = ["MUL0001", "MUL0002", "MUL0004"]
DATA_SHA = "96b6ff493a8294ffaf03021c024ef4aa2bfe92f980b84c029c81f2aa0f07f728"  # pragma: allowlist secret - publisher full original JSON identity


def verify_author(root):
    root = Path(root).resolve(strict=True)
    for name, identity in PINS.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("original author file escapes root")
        with path.open("rb") as stream:
            raw = stream.read(50_001)
        if (
            len(raw) > 50_000
            or hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != identity
        ):
            raise ValueError("original author Git blob mismatch: " + name)
    return root


def acquire_author(root, *, opener=urlopen):
    root = Path(root).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("use external server storage for original source")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, identity in PINS.items():
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/spokenwoz/LLM/dst/{name}"
        with opener(url, timeout=30) as response:
            raw = response.read(50_001)
        if (
            len(raw) > 50_000
            or hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != identity
        ):
            raise ValueError("original author download identity mismatch")
        with (root / name).open("xb") as stream:
            stream.write(raw)
    return verify_author(root)


def validate_selection(start=0, count=3, full=False, total=1000):
    if type(full) is not bool or type(start) is not int or type(count) is not int:
        raise ValueError("explicit boolean/full and integer dialogue range required")
    if full:
        if start != 0 or count != 3:
            raise ValueError("full cannot mix with an explicit range")
        return
    if start < 0 or count < 3 or start + count > total:
        raise ValueError("select a fixed range of at least3 complete dialogues")


def select_dialogues(data, *, start=0, count=3, full=False):
    validate_selection(start, count, full, len(data))
    if full:
        return dict(data)
    return {key: data[key] for key in list(data)[start : start + count]}


def selected_originals(data_root, *, start=0, count=3, full=False):
    path = Path(data_root) / "data.json"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if path.stat().st_size != 52_233_254 or digest.hexdigest() != DATA_SHA:
        raise ValueError("complete pinned original public-test JSON required")
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data) != 1000 or list(data)[:3] != IDS:
        raise ValueError("original public test ordering changed")
    return select_dialogues(data, start=start, count=count, full=full)


def expected_prompts(dialogues, author_root):
    """Independently reconstruct the pinned author's literals and exact loops."""
    source = (verify_author(author_root) / "dst_data.py").read_text(encoding="utf-8")
    constants = {}
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"prompt_domain", "domain_slot_description_dict"}
        ):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    prefix = constants["prompt_domain"]
    for slot, description in constants["domain_slot_description_dict"].items():
        prefix += " - Slot Name: " + slot + "; Slot Descrption: " + description + "\n "
    rows = []
    for identity, dialogue in dialogues.items():
        if not dialogue["log"] or len(dialogue["log"]) % 2:
            raise ValueError("complete original dialogue required")
        for index in range(0, len(dialogue["log"]), 2):
            content = prefix
            for position, turn in enumerate(dialogue["log"][: index + 1]):
                content += " USER: " if position % 2 == 0 else " SYSTEM: "
                content += turn["text"] + " \n "
            answer = {}
            for domain, groups in dialogue["log"][index + 1]["metadata"].items():
                for group, slots in groups.items():
                    for slot, value in slots.items():
                        if group == "semi" or slot != "booked":
                            answer[domain + "-" + slot] = value
            rows.append(
                {
                    "basename": identity + "-" + str(index),
                    "answer": answer,
                    "content": content + "STATE:",
                }
            )
    return rows


def prepare(data_root, author_root, output, *, start=0, count=3, full=False):
    author_root = verify_author(author_root)
    dialogues = selected_originals(data_root, start=start, count=count, full=full)
    output = Path(output).resolve()
    protected = [
        Path(data_root).resolve(),
        author_root,
        Path(__file__).resolve().parents[1],
    ]
    if any(
        output.is_relative_to(path) or path.is_relative_to(output) for path in protected
    ):
        raise ValueError("fresh external output must not overlap source roots")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    staged = output / "author"
    staged.mkdir()
    for name in PINS:
        (staged / name).write_bytes((author_root / name).read_bytes())
    (staged / "data.json").write_text(json.dumps(dialogues), encoding="utf-8")
    (output / "valListFile.json").write_text(
        "\n".join(dialogues) + "\n", encoding="utf-8"
    )
    original = (staged / "dst_data.py").read_text(encoding="utf-8")
    if original.count("f2.write(b)") != 2:
        raise ValueError("pinned intermediate write contract changed")
    compatible = original.replace("f2.write(b)", "f2.write(b)\nf2.flush()")
    script = staged / "dst_data.compat.py"
    script.write_text(compatible, encoding="utf-8")
    (output / "compatibility.diff").write_text(
        "".join(
            difflib.unified_diff(original.splitlines(True), compatible.splitlines(True))
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=staged,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (output / "prepare.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("original preparer failed; preserve output/prepare.log")
    expected = expected_prompts(dialogues, staged)
    actual = [
        json.loads(line)
        for line in (staged / "prompt_test_output_cleangpt.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if actual != expected:
        raise ValueError("original prepared prompts/references differ")
    report = {
        "source_id": "SPOKENWOZ",
        "status": "FIXED_ORIGINAL_DST_PROMPTS_PREPARED_NOT_EXECUTED",
        "dialogue_ids": list(dialogues),
        "original_turns": sum(len(row["log"]) for row in dialogues.values()),
        "original_requests": len(expected),
        "selection": {"start": start, "count": count, "full": full},
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (output / "preparation-public.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--start", type=int, default=0, help="Original dialogue index, not turn index"
    )
    parser.add_argument(
        "--count", type=int, default=3, help="Complete dialogues, at least3"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="All1000 original dialogues; no turn truncation",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-author", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        validate_selection(args.start, args.count, args.full)
    except ValueError as error:
        parser.error(str(error))
    if args.download_author:
        if args.author_root is None:
            parser.error("--author-root required")
        acquire_author(args.author_root)
        result = {
            "status": "ORIGINAL_AUTHOR_FILES_VERIFIED_NOT_EXECUTED",
            "model_calls": 0,
        }
    elif args.execute:
        if any(
            value is None for value in (args.author_root, args.data_root, args.output)
        ):
            parser.error("--author-root, --data-root and --output required")
        result = prepare(
            args.data_root,
            args.author_root,
            args.output,
            start=args.start,
            count=args.count,
            full=args.full,
        )
    else:
        result = {
            "status": "PLAN_ONLY_NO_DOWNLOAD",
            "repository": REPOSITORY,
            "revision": REVISION,
            "original_author_files": list(PINS),
            "selection": {"start": args.start, "count": args.count, "full": args.full},
            "original_dialogue_ids": IDS
            if not args.full and args.start == 0 and args.count == 3
            else None,
            "original_requests": 17782
            if args.full
            else (68 if args.start == 0 and args.count == 3 else None),
            "model_calls": 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
