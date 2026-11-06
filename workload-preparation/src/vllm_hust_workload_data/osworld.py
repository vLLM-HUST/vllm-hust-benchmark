"""Original OSWorld task packages; not rendered prompts or GUI execution."""

from copy import deepcopy
import argparse
from contextlib import closing, nullcontext
import json
from pathlib import Path
import re

from .assets import asset_path, verify_file


def original_task_packages(author_root, *, manifest):
    """Preserve original task order, setup, references and native asset identities.

    Callers must close a partially consumed iterator. The current task and task
    list are verified again at advance/close, matching other native packages.
    """
    if manifest.get("source_id") != "OSWORLD" or not manifest.get("revision"):
        raise ValueError("pinned original OSWorld source manifest required")
    files = {row["path"]: row for row in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("distinct original source paths required")
    catalog = "evaluation_examples/test_all.json"
    verify_file(author_root, files[catalog])
    groups = json.loads(asset_path(author_root, catalog).read_bytes())
    seen = set()
    for domain, identities in groups.items():
        if (
            not isinstance(domain, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", domain)
            or not isinstance(identities, list)
        ):
            raise ValueError("original domain/task list required")
        for identity in identities:
            if (
                not isinstance(identity, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", identity)
                or (domain, identity) in seen
            ):
                raise ValueError("distinct contained original task identities required")
            seen.add((domain, identity))
            path = f"evaluation_examples/examples/{domain}/{identity}.json"
            spec = files[path]
            verify_file(author_root, spec)
            task = json.loads(asset_path(author_root, path).read_bytes())
            if (
                task.get("id") != identity
                or not isinstance(task.get("instruction"), str)
                or ("config" in task and not isinstance(task["config"], list))
                or not isinstance(task.get("evaluator"), dict)
            ):
                raise ValueError("original task/setup/evaluator fields required")
            try:
                yield dict(
                    task_id=identity,
                    domain=domain,
                    source=dict(
                        source_id="OSWORLD",
                        repository=manifest["repository"],
                        revision=manifest["revision"],
                    ),
                    original_asset=deepcopy(spec),
                    input={
                        key: deepcopy(value)
                        for key, value in task.items()
                        if key != "evaluator"
                    },
                    outcome=dict(evaluator=deepcopy(task["evaluator"])),
                    native_execution_complete=False,
                )
            finally:
                verify_file(author_root, spec)
                verify_file(author_root, files[catalog])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        if args.output is None:
            parser.error("--write requires a fresh external output JSONL")
        output = args.output.resolve()
        source = args.author_root.resolve()
        if (
            args.output.is_symlink()
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError("output must be outside original source")
    count = 0
    with (
        args.output.open("x", encoding="utf-8") if args.write else nullcontext(None)
    ) as stream:
        with closing(
            original_task_packages(
                args.author_root, manifest=json.loads(args.manifest.read_text("utf-8"))
            )
        ) as rows:
            for row in rows:
                count += 1
                if stream is not None:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            dict(
                status="ORIGINAL_TASK_PACKAGES_VERIFIED_NOT_NATIVE_EXECUTION",
                records=count,
                written=args.write,
                native_execution_complete=False,
            )
        )
    )


if __name__ == "__main__":
    main()
