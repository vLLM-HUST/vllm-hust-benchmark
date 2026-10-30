#!/usr/bin/env python3
"""Prepare a fresh ARM CPU client and invoke the existing SWE-bench Pro runner.

No model deployment, host Docker access, old workspace copy or Controller needed.
The private work directory contains all downloads, client packages and results.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "src/vllm_hust_benchmark/data/swe_bench_pro_v1_lock.json"


RUNTIME_SOURCE_FILES = [
    "README.md",
    "pyproject.toml",
    "src/vllm_hust_benchmark/__init__.py",
    "src/vllm_hust_benchmark/_version.py",
    "src/vllm_hust_benchmark/swe_bench_pro.py",
    "src/vllm_hust_benchmark/swe_bench_pro_inputs.py",
    "src/vllm_hust_benchmark/swe_bench_pro_grading.py",
    "src/vllm_hust_benchmark/swe_bench_pro_timing.py",
    "src/vllm_hust_benchmark/swe_submodule_fixture.py",
    "src/vllm_hust_benchmark/swe_patch_state.py",
    "src/vllm_hust_benchmark/data/swe_bench_pro_v1_lock.json",
    "src/vllm_hust_benchmark/data/swe_bench_pro_bootstrap_lock.json",
    "docs/examples/swe-bench-pro/client-requirements.txt",
    "docs/examples/swe-bench-pro/single-agent.json",
    "scripts/prepare_swe_bench_pro_environment.py",
    "scripts/run_swe_bench_pro.py",
]


def source_repository(argv):
    """A single downloaded installer can fetch only its small runtime sources.

    Use GitHub's official API on networks where git/raw endpoints are blocked.
    Resolve the revision once, then download every file at that immutable commit.
    """
    if LOCK.is_file():
        return ROOT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-revision", default="main")
    parser.add_argument("--source-directory", type=Path)
    args, _ = parser.parse_known_args(argv)
    request_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "swe-bench-pro-bootstrap",
    }
    api = "https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark"
    revision_url = api + "/commits/" + urllib.parse.quote(args.source_revision, safe="")
    with urllib.request.urlopen(
        urllib.request.Request(revision_url, headers=request_headers), timeout=30
    ) as response:
        revision = json.load(response)["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("source revision did not resolve to an immutable Git commit")
    directory = (
        args.source_directory
        or Path.home() / ".local/share/swe-bench-pro/source" / revision
    )
    if not directory.is_absolute() or directory.is_symlink():
        raise ValueError("source-directory must be an absolute non-symlink directory")
    receipt_path = directory / "source-installation.json"
    if directory.exists():
        if (
            not receipt_path.is_file()
            or json.loads(receipt_path.read_text())["commit"] != revision
        ):
            raise ValueError(
                "source directory exists without this installer revision; refusing overwrite"
            )
        receipt = json.loads(receipt_path.read_text())
    else:
        directory.mkdir(parents=True, mode=0o700)
        receipt = {"commit": revision, "files": {}}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    trees = {}

    def tree_entries(tree):
        if tree not in trees:
            request = urllib.request.Request(
                api + "/git/trees/" + tree, headers=request_headers
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            if result.get("truncated"):
                raise ValueError("source Git tree is truncated")
            trees[tree] = {entry["path"]: entry for entry in result["tree"]}
        return trees[tree]

    for relative in RUNTIME_SOURCE_FILES:
        tree = revision
        for component in Path(relative).parts[:-1]:
            entry = tree_entries(tree)[component]
            if entry["type"] != "tree":
                raise ValueError("source parent is not a Git tree: " + relative)
            tree = entry["sha"]
        blob = tree_entries(tree)[Path(relative).name]
        if blob["type"] != "blob":
            raise ValueError("source is not a Git blob: " + relative)
        path = directory / relative
        if path.exists():
            if receipt["files"].get(relative) != sha256(path):
                raise ValueError("installed source changed: " + relative)
            verify_git_blob(path, blob)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = {
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": "swe-bench-pro-bootstrap",
        }
        for attempt in range(3):
            try:
                download(
                    api + "/contents/" + relative + "?ref=" + revision,
                    path,
                    headers=headers,
                    git_blob=blob,
                )
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
        receipt["files"][relative] = sha256(path)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("SOURCE_COMMIT " + revision, flush=True)
    return directory


def bundle_repository(url, checksum, revision, directory=None):
    """Install the published runtime without GitHub API calls or shared caches."""
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("source bundle requires an immutable source revision")
    if not re.fullmatch(r"[a-f0-9]{64}", checksum or ""):
        raise ValueError("source bundle requires its published SHA256")
    directory = (
        directory or Path.home() / ".local/share/swe-bench-pro/source" / revision
    )
    if not directory.is_absolute() or directory.is_symlink():
        raise ValueError("source-directory must be an absolute non-symlink directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    archive_path = directory.parent / (revision + ".tar.gz")
    download(url, archive_path, checksum)
    with tarfile.open(archive_path) as archive:
        entries = archive.getmembers()
        names = [entry.name for entry in entries]
        required = set(RUNTIME_SOURCE_FILES) | {"source-bundle.json"}
        if len(names) != len(set(names)) or set(names) != required:
            raise ValueError("source bundle file list differs from required runtime")
        if any(not entry.isfile() for entry in entries):
            raise ValueError("source bundle must contain regular files only")
        manifest = json.load(archive.extractfile("source-bundle.json"))
        if manifest["commit"] != revision or set(manifest["files"]) != set(
            RUNTIME_SOURCE_FILES
        ):
            raise ValueError("source bundle revision or manifest differs")
        data = {name: archive.extractfile(name).read() for name in RUNTIME_SOURCE_FILES}
        if any(
            hashlib.sha256(value).hexdigest() != manifest["files"][name]
            for name, value in data.items()
        ):
            raise ValueError("source bundle file SHA256 differs")
    receipt_path = directory / "source-installation.json"
    if directory.exists():
        if (
            not receipt_path.is_file()
            or json.loads(receipt_path.read_text()).get("commit") != revision
        ):
            raise ValueError("source directory exists without this installer revision")
        for name, checksum in manifest["files"].items():
            if sha256(directory / name) != checksum:
                raise ValueError("installed source changed: " + name)
        return directory
    directory.mkdir(mode=0o700)
    for name, value in data.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    receipt_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("SOURCE_COMMIT " + revision, flush=True)
    return directory


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_git_blob(path, blob):
    data = path.read_bytes()
    digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if len(data) != blob["size"] or digest != blob["sha"]:
        raise ValueError("source size or Git blob SHA differs: " + str(path))


def download(url, path, expected=None, headers=None, git_blob=None):
    if path.exists():
        if expected and sha256(path) == expected:
            return
        raise ValueError(
            "download destination already exists without matching hash: " + str(path)
        )
    temporary = path.with_suffix(path.suffix + ".partial")
    request_headers = {"User-Agent": "swe-bench-pro-bootstrap/1.0", **(headers or {})}
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with (
            urllib.request.urlopen(request, timeout=20) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    except OSError:
        curl = shutil.which("curl")
        if not curl or urllib.parse.urlsplit(url).scheme not in ("https", "http"):
            raise
        command = [
            curl,
            "--fail",
            "--location",
            "--retry",
            "2",
            "--retry-all-errors",
            "--connect-timeout",
            "10",
            "--max-time",
            "120",
            "--output",
            str(temporary),
        ]
        for name, value in request_headers.items():
            command.extend(["--header", name + ": " + value])
        subprocess.run(command + [url], check=True, timeout=400)
    if expected and sha256(temporary) != expected:
        raise ValueError("download SHA256 differs: " + str(path))
    if git_blob is not None:
        verify_git_blob(temporary, git_blob)
    temporary.replace(path)


def model_check(base, name, key_environment):
    url = urllib.parse.urlsplit(base)
    if (
        url.scheme not in ("http", "https")
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "model API must be an HTTP(S) URL without credentials/query/fragment"
        )
    if not url.path.rstrip("/").endswith("/v1"):
        raise ValueError("model API must include /v1")
    headers = {}
    if key_environment:
        if not os.environ.get(key_environment):
            raise ValueError(
                "missing model credential environment variable: " + key_environment
            )
        headers["Authorization"] = "Bearer " + os.environ[key_environment]
    with urllib.request.urlopen(
        urllib.request.Request(base.rstrip("/") + "/models", headers=headers),
        timeout=20,
    ) as response:
        models = json.load(response)["data"]
    # The provider prefix belongs to LiteLLM, not the server's served-model name.
    served = name.removeprefix("openai/")
    match = next((model for model in models if model["id"] == served), None)
    if not match:
        raise ValueError("requested model is absent from /v1/models")
    return {"model_name": name, "max_model_len": match.get("max_model_len")}


def main():
    global ROOT, LOCK
    if not LOCK.is_file():
        if platform.system() != "Linux" or platform.machine() not in (
            "aarch64",
            "arm64",
        ):
            raise ValueError("standalone installation requires Linux ARM64")
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument("--source-bundle-url")
        options.add_argument("--source-bundle-sha256")
        options.add_argument("--source-revision", default="main")
        options.add_argument("--source-directory", type=Path)
        bundle, _ = options.parse_known_args()
        ROOT = (
            bundle_repository(
                bundle.source_bundle_url,
                bundle.source_bundle_sha256,
                bundle.source_revision,
                bundle.source_directory,
            )
            if bundle.source_bundle_url
            else source_repository(sys.argv[1:])
        )
        LOCK = ROOT / "src/vllm_hust_benchmark/data/swe_bench_pro_v1_lock.json"
    artifact_lock = json.loads(
        (
            ROOT / "src/vllm_hust_benchmark/data/swe_bench_pro_bootstrap_lock.json"
        ).read_text()
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-revision",
        default="main",
        help="standalone installer source revision; resolved to one immutable commit",
    )
    p.add_argument(
        "--source-directory",
        type=Path,
        help="optional private source installation directory",
    )
    p.add_argument("--work", required=True, type=Path)
    p.add_argument(
        "--source-bundle-url",
        help="published small runtime bundle; avoids GitHub API rate limits",
    )
    p.add_argument(
        "--source-bundle-sha256", help="published SHA256 for the runtime bundle"
    )
    p.add_argument("--model-api", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--api-key-env")
    p.add_argument(
        "--dataset",
        type=Path,
        help="optional authorized local V1 parquet; otherwise download pinned public V1",
    )
    p.add_argument(
        "--dataset-base",
        default=os.environ.get("SWE_BENCH_PRO_DATASET_BASE", "https://huggingface.co"),
        help="explicit compatible dataset mirror",
    )
    p.add_argument("--docker", default=shutil.which("docker"))
    p.add_argument(
        "--docker-host",
        default=os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
    )
    p.add_argument("--crane", default=shutil.which("crane"))
    p.add_argument(
        "--crane-archive-url",
        default=artifact_lock["crane"]["linux_arm64_url"],
        help="official v0.20.3 ARM64 crane asset API URL or operator mirror",
    )
    p.add_argument(
        "--crane-archive-sha256",
        default=artifact_lock["crane"]["linux_arm64_sha256"],
        help="SHA256 of the ARM64 crane archive",
    )
    p.add_argument(
        "--image-mirror-prefix",
        default=os.environ.get("SWE_BENCH_PRO_IMAGE_MIRROR", ""),
    )
    p.add_argument(
        "--insecure-image-registry",
        action="store_true",
        default=os.environ.get("SWE_BENCH_PRO_IMAGE_REGISTRY_INSECURE", "false")
        == "true",
        help="explicit HTTP operator registry; HTTPS remains default",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume preparation in this exact bootstrap-owned work directory",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="run ONE real Agent after preparation; no automatic resubmission",
    )
    args = p.parse_args()
    work = args.work.expanduser().absolute()
    if not args.model_name.startswith("openai/"):
        p.error(
            "model-name must use openai/<served-model-name> for this OpenAI-compatible entry"
        )
    if not args.work.is_absolute() or work.is_symlink():
        p.error("work must be an absolute non-symlink private directory")
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        p.error("this entry requires Linux ARM64; it cannot run AMD64 images on ARM")
    if not args.docker:
        p.error(
            "Docker CLI missing: create an authorized CPU Docker executor first; see guide"
        )
    if not args.crane and not (args.crane_archive_url and args.crane_archive_sha256):
        p.error(
            "provide crane or an ARM64 release archive URL and its SHA256; see guide"
        )
    marker = work / "bootstrap-state.json"
    identity = {
        "repository": str(ROOT),
        "model_api": args.model_api,
        "model_name": args.model_name,
        "docker_host": args.docker_host,
        "dataset": None if not args.dataset else str(args.dataset.absolute()),
        "dataset_base": args.dataset_base,
        "image_mirror_prefix": args.image_mirror_prefix,
        "api_key_env": args.api_key_env,
        "image_registry_insecure": args.insecure_image_registry,
    }
    if work.exists():
        old = json.loads(marker.read_text()) if marker.is_file() else {}
        old_identity = old.get("identity", {})

        def comparable(value):
            return {
                k: v
                for k, v in value.items()
                if k not in ("image_mirror_prefix", "image_registry_insecure")
            }

        if not args.resume or comparable(old_identity) != comparable(identity):
            p.error(
                "work exists; only --resume with the same bootstrap identity is allowed"
            )
        if old_identity != identity and (work / "run").exists():
            p.error("cannot change image mirror after Agent submission")
        state = old
        if old_identity != identity:
            state.setdefault("network_changes", []).append(
                {"before": old_identity, "after": identity}
            )
            state["identity"] = identity
    else:
        work.mkdir(parents=True, mode=0o700)
        state = {"identity": identity, "completed": [], "status": "PREPARING"}
    os.chmod(work, 0o700)
    os.umask(0o077)

    def save():
        marker.write_text(json.dumps(state, indent=2) + "\n")

    save()
    env = dict(os.environ, DOCKER_HOST=args.docker_host, PYTHONPATH=str(ROOT / "src"))

    def command(step, command_args):
        if step in state["completed"]:
            return
        print("START " + step, flush=True)
        with (work / (step + ".log")).open("w") as log:
            result = subprocess.run(
                [str(x) for x in command_args],
                check=False,
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        if result.returncode:
            state.update(status="FAILED", failed_step=step)
            save()
            raise RuntimeError(
                "step failed: " + step + "; see " + str(work / (step + ".log"))
            )
        state["completed"].append(step)
        save()
        print("PASS " + step, flush=True)

    try:
        if shutil.disk_usage(work).free < 15 * 1024**3:
            raise ValueError(
                "at least 15 GiB free required for one image, archive and client; no disk quota is enlarged"
            )
        docker = subprocess.check_output(
            [args.docker, "info", "--format", "{{json .}}"],
            env=env,
            text=True,
            timeout=30,
        )
        info = json.loads(docker)
        if info.get("OSType") != "linux" or info.get("Architecture") not in (
            "aarch64",
            "arm64",
        ):
            raise ValueError("Docker daemon must be Linux ARM64")
        state["model"] = model_check(args.model_api, args.model_name, args.api_key_env)
        state["docker"] = {
            k: info.get(k)
            for k in ("ServerVersion", "Architecture", "DockerRootDir", "Images")
        }
        state.setdefault("initial_docker", state["docker"])
        save()
        lock = json.loads(LOCK.read_text())
        crane = args.crane
        if not crane:
            archive = work / "crane-arm64.tar.gz"
            download(
                args.crane_archive_url,
                archive,
                args.crane_archive_sha256,
                headers={"Accept": "application/octet-stream"},
            )
            binary = work / "bin/crane"
            binary.parent.mkdir(exist_ok=True)
            with tarfile.open(archive) as tar:
                member = tar.getmember("crane")
                if not member.isfile():
                    raise ValueError("crane release entry is not a regular file")
                binary.write_bytes(tar.extractfile(member).read())
            binary.chmod(0o700)
            crane = str(binary)
        dataset = work / "v1.parquet"
        if args.dataset:
            if sha256(args.dataset) != lock["dataset"]["parquet_sha256"]:
                raise ValueError("local dataset SHA256 differs from pinned V1")
            if not dataset.exists():
                shutil.copyfile(args.dataset, dataset)
        else:
            dataset_pin = lock["dataset"]
            base = args.dataset_base.rstrip("/")
            url = (
                base
                + "/datasets/"
                + dataset_pin["repository"]
                + "/resolve/"
                + dataset_pin["revision"]
                + "/data/v1/test-00000-of-00001.parquet"
            )
            download(url, dataset, dataset_pin["parquet_sha256"])
        if sha256(dataset) != lock["dataset"]["parquet_sha256"]:
            raise ValueError("prepared V1 dataset SHA256 differs")
        image_pin = artifact_lock["single_task_image"]
        image_reference = (
            args.image_mirror_prefix + image_pin["source"] + ":" + image_pin["tag"]
        )
        crane_command = [crane] + (
            ["--insecure"] if args.insecure_image_registry else []
        )
        observed_digest = subprocess.check_output(
            crane_command + ["digest", "--platform=linux/arm64", image_reference],
            env=env,
            text=True,
            timeout=45,
        ).strip()
        if observed_digest != image_pin["manifest_digest"]:
            raise ValueError(
                "task image is absent or differs from the frozen upstream digest; ask the platform operator to publish it before client installation"
            )
        state["preflight_image"] = {
            "reference": image_reference,
            "manifest_digest": observed_digest,
        }
        save()
        if "sources" not in state["completed"]:
            if (work / "sources").exists():
                raise ValueError(
                    "partial source preparation retained; inspect sources.log before using a new work directory"
                )
            command(
                "sources",
                [
                    sys.executable,
                    ROOT / "scripts/prepare_swe_bench_pro_environment.py",
                    "--root",
                    work / "sources",
                ],
            )
        client = work / "client"
        command("client", [sys.executable, "-m", "venv", client])
        python = client / "bin/python"
        requirements = ROOT / "docs/examples/swe-bench-pro/client-requirements.txt"
        command("dependencies", [python, "-m", "pip", "install", "-r", requirements])
        command(
            "agent",
            [
                python,
                "-m",
                "pip",
                "install",
                "-c",
                requirements,
                "-e",
                work / "sources/agent",
            ],
        )
        command("benchmark", [python, "-m", "pip", "install", "--no-deps", "-e", ROOT])
        with (work / "client-freeze-private.txt").open("w") as freeze:
            subprocess.run(
                [str(python), "-m", "pip", "freeze", "--all"],
                check=True,
                stdout=freeze,
                env=env,
                timeout=60,
            )
        settings_path = work / "settings-private.json"
        if not settings_path.exists():
            settings = json.loads(
                (ROOT / "docs/examples/swe-bench-pro/single-agent.json").read_text()
            )
            settings["run_id"] = "swe-pro-" + work.name
            settings["paths"].update(
                dataset_parquet=str(dataset),
                agent_source=str(work / "sources/agent"),
                official_v1_source=str(work / "sources/official-v1"),
                docker=args.docker,
                docker_host=args.docker_host,
                crane=crane,
            )
            settings["model"].update(
                model_name=args.model_name,
                api_base=args.model_api,
                api_key_env=args.api_key_env,
            )
            settings["image_mirror_prefix"] = args.image_mirror_prefix
            settings["image_registry_insecure"] = args.insecure_image_registry
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        elif state.get("failed_step") == "prepare" and not (work / "run").exists():
            settings = json.loads(settings_path.read_text())
            settings["image_mirror_prefix"] = args.image_mirror_prefix
            settings["image_registry_insecure"] = args.insecure_image_registry
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        inputs = work / "inputs"
        if (
            state.get("failed_step") == "prepare"
            and "prepare" not in state["completed"]
            and inputs.exists()
        ):
            if inputs.is_symlink():
                raise ValueError("preparation output must not be a symlink")
            attempt = len(list(work.glob("inputs-failed-*"))) + 1
            inputs.rename(work / f"inputs-failed-{attempt:03d}")
        runner = [python, "-m", "vllm_hust_benchmark.swe_bench_pro"]
        command(
            "check",
            runner + ["check", "--settings", settings_path, "--output", work / "check"],
        )
        command(
            "prepare",
            runner
            + ["prepare", "--settings", settings_path, "--output", work / "inputs"],
        )
        image_pin = artifact_lock["single_task_image"]
        selection = json.loads((work / "inputs/input-selection.json").read_text())
        if len(selection["tasks"]) != 1 or any(
            selection["tasks"][0][k] != image_pin[v]
            for k, v in [
                ("instance_id", "instance_id"),
                ("manifest_digest", "manifest_digest"),
                ("image_id", "config_digest"),
            ]
        ):
            raise ValueError(
                "introductory task image differs from pinned upstream manifest/config; no Agent submitted"
            )
        subprocess.run(
            [
                str(x)
                for x in runner
                + [
                    "verify-inputs",
                    "--settings",
                    settings_path,
                    "--inputs",
                    work / "inputs",
                    "--output",
                    work / "verification",
                ]
            ],
            env=env,
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        state["status"] = "READY"
        save()
        if args.run:
            if (work / "run").exists():
                raise ValueError(
                    "run already exists; refusing automatic Agent resubmission"
                )
            command(
                "run",
                runner
                + [
                    "run",
                    "--settings",
                    settings_path,
                    "--inputs",
                    work / "inputs",
                    "--output",
                    work / "run",
                ],
            )
            state["status"] = "RUN_RECORDED"
            state["result"] = json.loads((work / "run/batch-report.json").read_text())
            save()
            if state["result"]["submitted"] != 1:
                raise RuntimeError(
                    "real Agent did not submit; see the preserved task result, not a deployment PASS"
                )
        print(
            "STATUS " + state["status"] + "; private evidence: " + str(marker),
            flush=True,
        )
    except Exception as error:
        state.update(status="FAILED", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
