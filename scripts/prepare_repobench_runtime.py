"""Prepare an explicit RepoBench compatibility environment without model execution.

The author supplies an unpinned requirements list (including stdlib difflib),
not a lock. This profile is therefore a declared integration environment, not
an author environment reproduction. Reuses the platform torch/torch-npu stack
through explicit system-site inheritance; never installs into that stack.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_repobench_original_assets import verify_original

DEPENDENCIES = (
    "huggingface-hub==1.27.0",
    "numpy==1.26.4",
    "pyarrow==17.0.0",
    "pandas==3.0.5",
    "datasets==2.21.0",
    "fire==0.7.0",
    "codebleu==0.7.0",
    "fuzzywuzzy==0.18.0",
    "python-Levenshtein==0.27.1",
    "tree-sitter==0.22.3",
    "tree-sitter-python==0.21.0",
    "tree-sitter-java==0.21.0",
)


GRAMMAR_SOURCES = {
    "tree-sitter-python": "tree-sitter-python @ https://github.com/tree-sitter/tree-sitter-python/archive/0f9047c857ed0990931b1f899c7d3bf403703147.zip",  # pragma: allowlist secret - official v0.21.0 public source revision
    "tree-sitter-java": "tree-sitter-java @ https://github.com/tree-sitter/tree-sitter-java/archive/953abfc8bb3eb2f578e1f461edba4a9885f974b8.zip",  # pragma: allowlist secret - official v0.21.0 public source revision
}


GRAMMAR_ARCHIVES = {
    "tree-sitter-python": (
        "0f9047c857ed0990931b1f899c7d3bf403703147",  # pragma: allowlist secret - official public Git revision
        304741,
        "1f48b560fafa7cca4e3c28b16de6f0f42185cea0a1d2085126b68aeca8b22c5f",  # pragma: allowlist secret - original public archive checksum
    ),  # pragma: allowlist secret - official source revision and archive checksum
    "tree-sitter-java": (
        "953abfc8bb3eb2f578e1f461edba4a9885f974b8",  # pragma: allowlist secret - official public Git revision
        231500,
        "d3fc22ddd53ae7d674988d4367d60cc312a1c8dca5e18ca9af63469be658fb92",  # pragma: allowlist secret - original public archive checksum
    ),  # pragma: allowlist secret - official source revision and archive checksum
}


def prepare_grammar_sources(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, (revision, size, checksum) in GRAMMAR_ARCHIVES.items():
        path = root / f"{name}.zip"
        if not path.exists():
            temporary = root / f"{name}.zip.partial"
            if temporary.exists():
                raise ValueError(
                    "preserve partial grammar source; inspect before retry"
                )
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--max-time",
                    "120",
                    "--max-filesize",
                    str(size),
                    "--output",
                    str(temporary),
                    f"https://api.github.com/repos/tree-sitter/{name}/zipball/{revision}",
                ],
                check=True,
            )
            candidate = temporary
        else:
            candidate = path
        if candidate.is_symlink() or candidate.stat().st_size != size:
            raise ValueError("original grammar archive identity mismatch")
        with candidate.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                raise ValueError("original grammar archive checksum mismatch")
        if candidate != path:
            candidate.rename(path)


def commands(uv, platform_python, environment, grammar_source_dir=None):
    python = Path(environment) / "bin/python"
    requirements = []
    for spec in DEPENDENCIES:
        name = spec.split("==")[0]
        if grammar_source_dir and name in GRAMMAR_SOURCES:
            requirements.append(
                name
                + " @ "
                + (Path(grammar_source_dir).resolve() / f"{name}.zip").as_uri()
            )
        else:
            requirements.append(GRAMMAR_SOURCES.get(name, spec))
    return [
        [
            str(uv),
            "venv",
            "--python",
            str(platform_python),
            "--system-site-packages",
            str(environment),
        ],
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            *requirements,
        ],
    ]


def qualified_python(environment):
    environment = Path(environment).resolve()
    options = dict(
        line.lower().split("=", 1)
        for line in (environment / "pyvenv.cfg").read_text().splitlines()
        if "=" in line
    )
    options = {key.strip(): value.strip() for key, value in options.items()}
    if options.get("include-system-site-packages") != "true":
        raise ValueError(
            "this declared profile requires explicit platform site inheritance"
        )
    python = environment / "bin/python"
    if not python.is_file():
        raise ValueError("Linux venv Python missing")
    return python


def verify_environment(environment, source):
    python = qualified_python(environment)
    # No generated code, network request, model loading or accelerator calls.
    code = """
import importlib.metadata as m,json,pathlib,sys
source=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(source))
import run
from evaluation import metrics
assert pathlib.Path(run.__file__).resolve()==source/'run.py'
assert pathlib.Path(metrics.__file__).resolve()==source/'evaluation/metrics.py'
deps=json.loads(sys.argv[2])
versions={}
for spec in deps:
    name,version=spec.split('==')
    versions[name]=m.version(name)
    assert versions[name]==version,(name,versions[name],version)
for name in ['torch','torch-npu','transformers']:
    versions[name]=m.version(name)
assert run.get_first_line_not_comment('# comment\\nx = 1','python')=='x = 1'
assert run.get_first_line_not_comment('// comment\\nint x = 1;','java')=='int x = 1;'
scores={}
for lang,text in [('python','def add(a,b):\\n    return a+b'),('java','class A { int f(int a) { return a+1; } }')]:
    scores[lang]={'em':metrics.exact_match_score([text],[text]),'es':metrics.edit_similarity_score([text],[text]),'codebleu':metrics.codebleu_score([text],[text],lang)}
    assert scores[lang]['em']==1 and scores[lang]['es']==100 and 0<=scores[lang]['codebleu']<=1
print(json.dumps({'status':'ORIGINAL_IMPORT_AND_SCORER_FIXTURES_VERIFIED_NOT_NATIVE_EXECUTION','installed_versions':versions,'scorer_fixture_scores':scores,'model_calls':0,'source_checkbox_eligible':False}))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(python), "-c", code, str(source), json.dumps(DEPENDENCIES)],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        result.check_returncode()
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--platform-python", type=Path, required=True)
    parser.add_argument(
        "--grammar-source-dir",
        type=Path,
        help="External fixed official archives; acquire missing archives via author API, verify exact hashes and build ARM wheels locally",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument(
        "--install-dependencies",
        action="store_true",
        help="Explicit dependency-install resume in an existing matching venv; never writes the platform stack",
    )
    args = parser.parse_args()
    planned = commands(
        args.uv, args.platform_python, args.environment, args.grammar_source_dir
    )
    if not args.execute and not args.verify_only and not args.install_dependencies:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "commands": planned,
                    "system_site_inheritance": True,
                    "author_lock_reproduction": False,
                    "model_calls": 0,
                }
            )
        )
        return
    verify_original(args.source_root, args.archive)
    if args.environment.resolve().is_relative_to(args.source_root.resolve()):
        raise ValueError("environment must remain outside publisher source")
    if args.grammar_source_dir and (args.execute or args.install_dependencies):
        repository = Path(__file__).resolve().parents[1]
        if (
            repository / ".git"
        ).exists() and args.grammar_source_dir.resolve().is_relative_to(repository):
            raise ValueError("original grammar archives must remain external")
        prepare_grammar_sources(args.grammar_source_dir)
    if args.execute:
        if args.environment.exists():
            raise ValueError("fresh environment required; preserve previous state")
        for command in planned:
            subprocess.run(command, check=True)
    elif args.install_dependencies:
        qualified_python(args.environment)
        subprocess.run(planned[1], check=True)
    result = verify_environment(args.environment, args.source_root)
    subprocess.run(
        [
            str(args.uv),
            "pip",
            "check",
            "--python",
            str(qualified_python(args.environment)),
        ],
        check=True,
    )
    result["venv_dependency_check_exit_code"] = 0
    result["platform_inheritance_not_author_lock_reproduction"] = True
    verify_original(args.source_root, args.archive)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
