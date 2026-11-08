"""Fresh verified original Agent software with two disclosed local runtime ports."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/mlagentbench-author-source.json"
OPTIONAL_IMPORT = b"from MLAgentBench.agents.agent_langchain  import LangChainAgent"
RUN = b"    run(getattr(sys.modules[__name__], args.agent_type), args)"


def port_runner(raw):
    if raw.count(OPTIONAL_IMPORT) != 1 or raw.count(RUN) != 1:
        raise ValueError("exact pinned original runner import/dispatch required")
    raw = raw.replace(
        OPTIONAL_IMPORT,
        b"# Local port: optional LangChain imports only when explicitly selected.",
    )
    return raw.replace(
        RUN,
        b'    if args.agent_type == "LangChainAgent":\n'
        b"        from MLAgentBench.agents.agent_langchain import LangChainAgent\n"
        + RUN,
    )


def prepare(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_AGENT_OR_MODEL_EXECUTION",
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    author = verify_source(args.author_root, PINS)
    output = args.output.resolve()
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in [ROOT, author]
    ):
        raise ValueError(
            "fresh external port workspace separate from originals required"
        )
    pins = json.loads(PINS.read_text("utf-8"))
    files = {
        item["path"]: (author / item["path"]).read_bytes() for item in pins["files"]
    }
    originals = {
        name: files[name] for name in ["MLAgentBench/LLM.py", "MLAgentBench/runner.py"]
    }
    files["MLAgentBench/runner.py"] = port_runner(originals["MLAgentBench/runner.py"])
    files["MLAgentBench/LLM.py"] = (
        (ROOT / "scripts/mlagentbench_local_llm_binding.py").read_text("utf-8").encode()
    )
    for name in ["mlagentbench_local_chat.py", "appworld_model_transport.py"]:
        files[name] = (ROOT / "scripts" / name).read_text("utf-8").encode()
    for name, raw in originals.items():
        files["originals/" + name + ".source"] = raw
        diff = difflib.unified_diff(
            raw.decode().splitlines(True),
            files[name].decode().splitlines(True),
            fromfile="original/" + name,
            tofile="local-port/" + name,
        )
        files["ports/" + name + ".diff"] = "".join(diff).encode()
    if args.action == "prepare":
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, raw in files.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
    for name, raw in files.items():
        path = output / name
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(output)
            or path.read_bytes() != raw
        ):
            raise ValueError(
                "ported workspace differs from declared original/software ports"
            )
    actual = {
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if actual != set(files):
        raise ValueError("unexpected port workspace files")
    report.update(
        status="ORIGINAL_AGENT_SOFTWARE_AND_DISCLOSED_PORTS_VERIFIED_NOT_EXECUTED",
        original_files=len(pins["files"]),
        ported_original_files=list(originals),
        task_or_prompt_modified=False,
        port_sha256={
            name: hashlib.sha256(files[name]).hexdigest() for name in originals
        },
    )
    return report


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
