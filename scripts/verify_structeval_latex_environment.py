"""Verify software only; no source tasks, generated TeX, models or VQA."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess


def verify():
    if metadata.version("pdf2image") != "1.17.0":
        raise ValueError("observed pdf2image version differs")
    tools = {}
    for name in ("pdflatex", "lualatex", "pdftoppm", "kpsewhich"):
        path = shutil.which(name)
        if not path:
            raise ValueError(f"missing original compiler/converter dependency: {name}")
        tools[name] = {
            "path": path,
            "observed_binary_sha256": hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest(),
        }
    tex_files = {}
    for name in ("standalone.cls", "tikz.sty", "pgfplots.sty", "fontspec.sty"):
        result = subprocess.run(
            [tools["kpsewhich"]["path"], name],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        path = Path(result.stdout.strip())
        if not path.is_file():
            raise ValueError(f"original TeX package unavailable: {name}")
        tex_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "status": "SOFTWARE_IDENTITIES_VERIFIED_NATIVE_QUALIFICATION_PENDING",
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "pdf2image": metadata.version("pdf2image"),
        "tools": tools,
        "original_tex_packages_observed_sha256": tex_files,
        "author_environment_lock_reproduced": False,
        "original_tasks_executed": 0,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
