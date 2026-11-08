"""Read the first fixed annotation from each of three author papers."""

from pathlib import Path
import re

from .assets import asset_path, verify_file
from .registry import source_spec


START = re.compile(r'^\s*#\s*<paper2code\s+name="([^"]+)">\s*$')
END = re.compile(r'^\s*#\s*</paper2code\s+name="([^"]+)">\s*$')


def read_annotation(lines):
    start, marker = next(
        (index, match)
        for index, line in enumerate(lines)
        if (match := START.fullmatch(line))
    )
    name = marker[1]
    end = next(
        index
        for index in range(start + 1, len(lines))
        if (match := END.fullmatch(lines[index])) and match[1] == name
    )
    reference = lines[start + 1 : end]
    indent = min(
        len(line) - len(line.lstrip(" \t")) for line in reference if line.strip()
    )
    code_lines = sum(bool(line.partition("#")[0].strip()) for line in reference)
    # The author Code model preserves the embedded newlines of its placeholder strings.
    replacement = [
        " " * indent + f'# TODO: Implement block "{name}"\n',
        " " * indent + f"# Approximately {code_lines} line(s) of code.\n",
    ]
    masked = "\n".join(
        line
        for line in [*lines[:start], *replacement, *lines[end + 1 :]]
        if not (START.fullmatch(line) or END.fullmatch(line))
    )
    return name, start, masked, "\n".join(reference)


def original_rows(path):
    path = Path(path).resolve(strict=True)
    root = path.parents[2]
    relative = path.relative_to(root).as_posix()
    files = source_spec("RESEARCHCODEBENCH")["files"]
    target = next(
        (item for item in files if item["role"] == "data" and item["path"] == relative),
        None,
    )
    if target is None:
        raise ValueError("expected one of the three pinned annotated source files")
    verify_file(root, target)
    paper_dir = path.parent.relative_to(root).as_posix()
    (paper,) = [
        item
        for item in files
        if item["path"].startswith(f"{paper_dir}/paper2code_paper.")
    ]
    evaluation = [
        verify_file(root, item)
        for item in files
        if item["path"].startswith(f"{paper_dir}/")
        and Path(item["path"]).name
        in {"paper2code_test.py", "model_ref.py", "Synthetic_ref.py"}
    ]
    verify_file(root, paper)
    name, start, masked, reference = read_annotation(
        path.read_text(encoding="utf-8").splitlines()
    )
    yield {
        "task_id": f"{path.parent.name}:{path.name}:{start}",
        "annotation": name,
        "paper_text": asset_path(root, paper["path"]).read_text(encoding="utf-8"),
        "masked_code": masked,
        "reference_code": reference,
        "evaluation_assets": evaluation,
    }
