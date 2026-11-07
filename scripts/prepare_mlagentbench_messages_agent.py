"""Independent native-Messages port; historical Chat entry/receipts stay intact."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import prepare_mlagentbench_local_agent as chat

ROOT = chat.ROOT
PINS = chat.PINS
PROFILE = {
    "protocol": "original_anthropic_messages_with_explicit_local_model",
    "max_tokens_to_sample": 2000,
    "temperature": 0.5,
    "stop_sequences": ["\n\nHuman:", "Observation:"],
    "edit_max_tokens": 4000,
    "fast_temperature": 0.01,
    "independent_of_historical_chat_results": True,
    "response_adapter": "First real text block; ignore separately typed reasoning blocks without changing text or requests",
}


def expected_files(root):
    pins = json.loads(PINS.read_text("utf-8"))
    files = {}
    for row in pins["files"]:
        name = row["path"]
        if name in {"MLAgentBench/LLM.py", "MLAgentBench/runner.py"}:
            backup = root / ("originals/" + name + ".source")
            original = backup.read_bytes()
            if (
                backup.is_symlink()
                or not backup.resolve().is_relative_to(root)
                or hashlib.sha256(original).hexdigest() != row["sha256"]
            ):
                raise ValueError("retained pinned original differs")
            files["originals/" + name + ".source"] = original
            raw = (
                chat.port_runner(original)
                if name.endswith("runner.py")
                else (ROOT / "scripts/mlagentbench_local_messages_binding.py")
                .read_text("utf-8")
                .encode()
            )
            files[name] = raw
            files["ports/" + name + ".diff"] = "".join(
                difflib.unified_diff(
                    original.decode().splitlines(True),
                    raw.decode().splitlines(True),
                    fromfile="original/" + name,
                    tofile="local-port/" + name,
                )
            ).encode()
        else:
            path = root / name
            raw = path.read_bytes()
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root)
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
            ):
                raise ValueError("original task/tool/Agent differs")
            files[name] = raw
    for name in [
        "mlagentbench_local_chat.py",
        "appworld_model_transport.py",
        "mlagentbench_local_messages.py",
    ]:
        files[name] = (ROOT / "scripts" / name).read_text("utf-8").encode()
    files["messages-profile.json"] = json.dumps(PROFILE, indent=2).encode()
    return files


def verify_messages_port(root):
    root = root.resolve(strict=True)
    files = expected_files(root)
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    dataset_link = root / "MLAgentBench/benchmarks/imdb/env/imdb"
    if dataset_link.is_symlink():
        if dataset_link.readlink() != Path("/hf/dataset"):
            raise ValueError(
                "only the original declared IMDb asset binding is accepted"
            )
        actual.remove(dataset_link.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError("unexpected file in independently prepared Messages workspace")
    for name, raw in files.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.read_bytes() != raw
        ):
            raise ValueError("Messages port differs from disclosed implementation")


def prepare(args):
    if args.action == "plan":
        return dict(
            status="PLAN_ONLY_NO_SOURCE_OR_MODEL_EXECUTION",
            profile=PROFILE,
            source_checkbox_eligible=False,
        )
    if args.action == "prepare":
        chat.prepare(
            SimpleNamespace(
                action="prepare", author_root=args.author_root, output=args.output
            )
        )
        files = expected_files(args.output.resolve())
        for name, raw in files.items():
            path = args.output / name
            if not path.exists() or path.read_bytes() != raw:
                path.write_bytes(
                    raw
                )  # Only this freshly created, verified port, never author source.
    verify_messages_port(args.output)
    return dict(
        status="PINNED_ORIGINAL_SOFTWARE_AND_INDEPENDENT_MESSAGES_PORT_VERIFIED",
        profile=PROFILE,
        source_checkbox_eligible=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
