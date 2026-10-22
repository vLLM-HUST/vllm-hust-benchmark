"""Build pinned ARM PDDL runtime wheels without editing runtime/task Python.

Build-only metadata adaptations, not an official Docker-equivalence claim. No
game tasks, model calls, data download or research observation is performed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import zipfile

ARCHIVES = {
    "fast-downward": "8d8fe8fb41e370dc1e442cd50592dc0d99243aacbcb82e8c83fec89e89f0ac05",  # pragma: allowlist secret - official source checksum
    "textworld": "bfbac893fec147297bbb1e07e51adb4b14adcc579db823515e42c801b8e64e19",  # pragma: allowlist secret - official source checksum
    "alfworld": "5e640152e9c2f3fb8534c8918c4f6e0eabe954b13b10635d551620fe89ad9a0a",  # pragma: allowlist secret - official source checksum
}


def adapt_packaging(component, arguments):
    """Alter build hooks/dependency metadata only, not a Python source file."""
    arguments = dict(arguments)
    if component == "textworld":
        if set(arguments.get("cmdclass", {})) != {"build_py", "install", "develop"}:
            raise ValueError("fixed TextWorld packaging hooks differ")
        arguments["cmdclass"] = {}  # Optional Inform7/Glulx executable setup.
    elif component == "alfworld":
        dependencies = []
        for requirement in arguments.get("install_requires", []):
            requirement = requirement.strip()
            if not requirement or any(
                key in requirement for key in ("numpy", "torch", "transformers")
            ):
                continue  # Same replacements as the pinned official Dockerfile.
            if requirement.startswith("opencv-python"):
                requirement = "opencv-python==4.11.0.86"
            elif requirement.startswith("fast-downward @"):
                requirement = "fast-downward==20.6"
            elif requirement.startswith("textworld @"):
                requirement = "textworld==1.3.2"
            dependencies.append(requirement)
        arguments["install_requires"] = dependencies + [
            "numpy<2",
            "torch<2",
            "torchvision<0.15",
            "transformers<4.48.0",
            "Werkzeug<3.0",
            "blis<=1.2.0",
            "jsonlines",
        ]
    elif component != "fast-downward":
        raise ValueError("unsupported runtime component")
    return arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downward-archive", type=Path, required=True)
    parser.add_argument("--textworld-archive", type=Path, required=True)
    parser.add_argument("--alfworld-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        sys.version_info[:2] != (3, 9)
        or platform.system() != "Linux"
        or platform.machine() != "aarch64"
    ):
        parser.error("this recipe is validated for Linux aarch64 Python 3.9 only")
    archives = {
        "fast-downward": args.downward_archive.resolve(),
        "textworld": args.textworld_archive.resolve(),
        "alfworld": args.alfworld_archive.resolve(),
    }
    for component, path in archives.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVES[component]:
            raise ValueError("fixed source archive identity mismatch: " + component)
    output = args.output.resolve()
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "wheels").mkdir()
    (output / "tmp").mkdir()
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    os.environ["TMPDIR"] = str(output / "tmp")
    os.environ["PIP_CACHE_DIR"] = str(output / "pip-cache")
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "2"
    import setuptools

    setup = setuptools.setup
    prior_argv, prior_cwd = sys.argv[:], Path.cwd()
    receipt = {
        "status": "BUILD_STARTED",
        "scope": "PDDL_RUNTIME_BUILD_ONLY",
        "original_task_code_changed": False,
        "model_calls": 0,
        "native_tasks_executed": 0,
        "official_container_equivalence_claim": False,
        "unsupported_paths": ["Inform7 compilation", "Glulx games", "visual THOR"],
        "components": {},
    }
    try:
        for component, path in archives.items():
            target = output / component
            with zipfile.ZipFile(path) as bundle:
                for name in bundle.namelist():
                    if not (target / name).resolve().is_relative_to(output):
                        raise ValueError("source archive path escape")
                bundle.extractall(target)
            source = next(path for path in target.iterdir() if path.is_dir())
            os.chdir(source)
            before = set((output / "wheels").glob("*.whl"))

            def adapted_setup(**arguments):
                return setup(**adapt_packaging(component, arguments))

            setuptools.setup = adapted_setup
            sys.argv = [
                str(source / "setup.py"),
                "bdist_wheel",
                "--dist-dir",
                str(output / "wheels"),
            ]
            runpy.run_path(str(source / "setup.py"), run_name="__main__")
            wheels = set((output / "wheels").glob("*.whl")) - before
            if len(wheels) != 1:
                raise ValueError("expected one wheel from fixed runtime source")
            wheel = wheels.pop()
            verified = 0
            if component in {"textworld", "alfworld"}:
                with zipfile.ZipFile(wheel) as bundle:
                    for name in bundle.namelist():
                        if name.startswith(component + "/") and name.endswith(".py"):
                            if bundle.read(name) != (source / name).read_bytes():
                                raise ValueError(
                                    "packaging changed an original runtime Python file"
                                )
                            verified += 1
                if verified == 0:
                    raise ValueError("runtime source verification covered zero files")
            receipt["components"][component] = {
                "wheel": wheel.name,
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "source_archive_sha256": ARCHIVES[component],
                "original_runtime_python_files_verified": verified,
            }
        receipt["status"] = "WHEELS_BUILT_AND_RUNTIME_SOURCE_VERIFIED"
    except BaseException as error:
        receipt.update(status="BUILD_FAILED", error_type=type(error).__name__)
        raise
    finally:
        setuptools.setup = setup
        sys.argv = prior_argv
        os.chdir(prior_cwd)
        (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
