"""Observe the pinned official grader without changing its scoring or patch."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

APPLY = "git apply -v /workspace/patch.diff"
TEST_MARKER = "# run test and save stdout and stderr to separate files"
HELPER = "hust_patch_state.py"


def instrument_entryscript(script):
    lines = script.splitlines()
    if lines.count(APPLY) != 1 or lines.count(TEST_MARKER) != 1:
        raise ValueError("Unsupported official entryscript layout")
    if lines.index(APPLY) >= lines.index(TEST_MARKER):
        raise ValueError("Official apply/test order changed")
    if "hust-patch-" in script:
        raise ValueError("Entry script already instrumented")

    def capture(stage):
        return (
            f"python /workspace/{HELPER} {stage} /workspace/patch.diff "
            f"/workspace/hust-patch-{stage}.json"
        )

    result = []
    for line in lines:
        if line == APPLY:
            result.extend(
                [
                    capture("before"),
                    APPLY + " > /workspace/hust-patch-apply.stdout "
                    "2> /workspace/hust-patch-apply.stderr",
                    "hust_patch_apply_exit=$?",
                    'printf "%s\\n" "$hust_patch_apply_exit" '
                    "> /workspace/hust-patch-apply.exit",
                    capture("after"),
                ]
            )
        else:
            if line == TEST_MARKER:
                result.append(capture("graded"))
            result.append(line)
    return "\n".join(result) + "\n"


def install(module):
    original_entry = module.create_entryscript
    original_files = module.assemble_workspace_files
    helper = Path(__file__).with_name("swe_patch_state.py").read_text()

    def entry(sample):
        return instrument_entryscript(original_entry(sample))

    def files(*args, **kwargs):
        items, script = original_files(*args, **kwargs)
        if HELPER in items:
            raise ValueError("Reserved evidence helper name already present")
        return {**items, HELPER: helper}, script

    module.create_entryscript = entry
    module.assemble_workspace_files = files


def summarize_application(workspace):
    """Keep application evidence separate from the unmodified official score."""
    workspace = Path(workspace)
    value = {"status": "unattested", "state_capture": "incomplete"}
    try:
        exit_code = int((workspace / "hust-patch-apply.exit").read_text().strip())
        value["apply_exit_code"] = exit_code
        if exit_code != 0:
            value["status"] = "apply_failed"
        states = {
            stage: json.loads((workspace / f"hust-patch-{stage}.json").read_text())
            for stage in ("before", "after", "graded")
        }
        if any(
            s.get("status") != "captured" or s.get("stage") != stage
            for stage, s in states.items()
        ):
            raise ValueError("Incomplete state capture")
        patch_sha = hashlib.sha256((workspace / "patch.diff").read_bytes()).hexdigest()
        if any(s["patch_sha256"] != patch_sha for s in states.values()):
            raise ValueError("Effective patch identity mismatch")
        if any(
            set(s["files"]) != set(states["before"]["files"]) for s in states.values()
        ):
            raise ValueError("Affected file set changed")
        evidence = {}
        for name in [
            "hust-patch-apply.exit",
            "hust-patch-apply.stdout",
            "hust-patch-apply.stderr",
            *[f"hust-patch-{s}.json" for s in states],
        ]:
            data = (workspace / name).read_bytes()
            evidence[name] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        value.update(
            status="applied" if exit_code == 0 else "apply_failed",
            state_capture="complete",
            apply_exit_code=exit_code,
            effective_patch_sha256=patch_sha,
            evidence=evidence,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        value["error_type"] = type(exc).__name__
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-script", required=True)
    parser.add_argument("--official-sha256", required=True)
    args, remaining = parser.parse_known_args()
    script = Path(args.official_script).resolve()
    if hashlib.sha256(script.read_bytes()).hexdigest() != args.official_sha256:
        raise ValueError("Official grader source changed")
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("hust_pinned_official_grader", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install(module)
    sys.argv = [str(script), *remaining]
    module.main()


if __name__ == "__main__":
    main()
