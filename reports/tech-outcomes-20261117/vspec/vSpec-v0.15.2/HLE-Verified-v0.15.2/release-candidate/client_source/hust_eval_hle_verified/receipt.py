"""Capture client-side evidence about a server that is already running.

This does NOT start or touch the server. It records what a client can legitimately observe or what the person
running the server hands over:
  * GET /v1/models (served name, max_model_len),
  * optionally GET /metrics,
  * a frozen copy of the server's startup log and actual argv (supplied as files),
  * the expected profile, its hash, and the comparison from config_check,
  * the client's own environment.
"""
from __future__ import annotations

import datetime
import json
import platform
import shutil
import socket
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Optional

from . import __version__, config_check
from .hashing import sha256_file

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url: str, timeout: int = 30, api_key: Optional[str] = None) -> bytes:
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read()


def _pkg(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def capture(run_dir, base_url: str, profile: Optional[str] = None, startup_log: Optional[str] = None,
            server_argv: Optional[str] = None, image_id: Optional[str] = None, with_metrics: bool = True,
            api_key: Optional[str] = None) -> dict:
    run = Path(run_dir)
    rc = run / "receipt"
    if (rc / "models.json").exists():
        raise FileExistsError("receipt already captured in %s; use a new run dir" % run)
    rc.mkdir(parents=True, exist_ok=True)

    models = json.loads(_get(base_url.rstrip("/") + "/v1/models", api_key=api_key).decode())
    (rc / "models.json").write_text(json.dumps(models, indent=1))
    if with_metrics:
        try:
            (rc / "metrics_at_capture.txt").write_bytes(_get(base_url.rstrip("/") + "/metrics", api_key=api_key))
        except Exception as e:  # noqa: BLE001 - /metrics is optional
            (rc / "metrics_at_capture.txt").write_text("UNAVAILABLE: %r\n" % (e,))

    log_text = None
    if startup_log:
        if Path(startup_log).resolve() != (rc / "startup.log").resolve():
            shutil.copy(startup_log, rc / "startup.log")  # frozen copy
        log_text = (rc / "startup.log").read_text(errors="replace")
    argv = None
    if server_argv:
        argv = config_check.load_argv(server_argv)
        (rc / "argv_actual.json").write_text(json.dumps(argv, indent=1))

    result = None
    if profile:
        if Path(profile).resolve() != (run / "profile.json").resolve():
            shutil.copy(profile, run / "profile.json")
        (rc / "profile.sha256").write_text("%s  profile.json\n" % sha256_file(run / "profile.json"))
        prof = json.loads((run / "profile.json").read_text())
        problems = config_check.profile_problems(prof)
        if problems:
            raise ValueError("profile has undecided fields: %s" % problems)
        result = config_check.verify(prof, argv, log_text, models)
        (rc / "effective.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        if not result["ok"]:
            (rc / "FAILED.txt").write_text("Profile mismatch: inspect effective.json\n")

    client = {
        "tool": "hust-eval-hle-verified %s" % __version__,
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_url": base_url,
        "client_host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "client_packages": {p: _pkg(p) for p in ("pyarrow", "transformers", "vllm")},
        "image_id": image_id or "NOT_PROVIDED",
        "note": "image_id is supplied by the operator; a client cannot read it from the server",
        "startup_log_provided": bool(startup_log),
        "server_argv_provided": bool(server_argv),
        "profile_provided": bool(profile),
    }
    (rc / "client.json").write_text(json.dumps(client, indent=1))
    return {"models": models, "effective": result, "client": client}
