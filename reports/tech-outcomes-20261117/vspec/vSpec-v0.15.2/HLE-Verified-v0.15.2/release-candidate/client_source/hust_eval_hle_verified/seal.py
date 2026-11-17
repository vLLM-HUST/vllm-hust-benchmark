"""Bind evidence to a result table by SHA-256, in two non-circular steps.

A result table cannot contain its own hash, so:
  1. evidence: hash every file in the run dir EXCEPT results/ and the seal files -> evidence_bundle_sha256.
               Put that value in the table's Provenance sheet.
  2. final:    hash the finished table, write MANIFEST.json (bundle hash + table hash), append a ledger line.
               Keep MANIFEST.json's own SHA-256 OUTSIDE the run dir (ledger / git / chat): that is the anchor.
  3. check:    re-verify a sealed run.

This is a plain SHA-256 listing. It is NOT RFC 8785 canonical JSON and NOT signed; do not call it a signed
evidence bundle.

The tool does not start or stop servers, so it cannot know whether a server is still appending to a log. Copy
any server log into the run dir (a frozen copy) BEFORE sealing.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from . import __version__
from .hashing import bundle_hash, hash_tree, sha256_file

SEAL_FILES = ("EVIDENCE_BUNDLE.json", "MANIFEST.json")
RESULTS_DIR = "results"


class SealError(Exception):
    pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def collect(run_dir) -> dict:
    return hash_tree(run_dir, exclude_top=(RESULTS_DIR,), exclude_files=SEAL_FILES)


def seal_evidence(run_dir) -> dict:
    run = Path(run_dir)
    if not run.is_dir():
        raise SealError("run dir not found: %s" % run)
    if any(run.rglob(".active.lock")):
        raise SealError("active client lock exists; finish requests before sealing")
    if any(run.rglob("FAILED.txt")):
        raise SealError("run has receipt/FAILED.txt - refusing to seal a failed run")
    out = run / "EVIDENCE_BUNDLE.json"
    if out.exists():
        raise SealError("EVIDENCE_BUNDLE.json already exists; a run is sealed once")
    files = collect(run)
    bundle = {"tool": "hust-eval-hle-verified %s" % __version__, "run_dir": str(run), "sealed_utc": _now(),
              "evidence_bundle_sha256": bundle_hash(files), "files": files}
    out.write_text(json.dumps(bundle, indent=1))
    return bundle


def seal_final(run_dir, table, ledger: Optional[str] = None,
               levels: str = "D=documented command; R=result files observed; S=environment snapshot") -> dict:
    run = Path(run_dir)
    eb = run / "EVIDENCE_BUNDLE.json"
    if not eb.exists():
        raise SealError("run `evidence` first")
    bundle = json.loads(eb.read_text())
    if bundle_hash(collect(run)) != bundle["evidence_bundle_sha256"]:
        raise SealError("evidence files changed after the `evidence` step; refusing to bind")
    table = Path(table)
    if not table.is_file():
        raise SealError("table not found: %s" % table)
    out = run / "MANIFEST.json"
    if out.exists():
        raise SealError("MANIFEST.json already exists")

    def opt(rel):
        p = run / rel
        return sha256_file(p) if p.is_file() else None

    man = {
        "tool": "hust-eval-hle-verified %s" % __version__,
        "run_id": run.name,
        "profile_sha256": opt("profile.json"),
        "evidence_bundle_sha256": bundle["evidence_bundle_sha256"],
        "effective_json_sha256": opt("receipt/effective.json"),
        "startup_log_sha256": opt("receipt/startup.log"),
        "table_path": table.name,
        "table_sha256": sha256_file(table),
        "sealed_utc": _now(),
        "evidence_levels": levels,
        "note": "plain SHA-256 listing; not RFC 8785 canonical, not signed",
    }
    out.write_text(json.dumps(man, indent=1))
    msha = sha256_file(out)
    if ledger:
        with open(ledger, "a") as f:
            f.write(json.dumps({"run_id": run.name, "manifest_sha256": msha, "table_sha256": man["table_sha256"],
                                "sealed_utc": man["sealed_utc"]}) + "\n")
    return {"table_sha256": man["table_sha256"], "manifest_sha256": msha, "manifest": man}


def seal_check(run_dir, table: Optional[str] = None) -> dict:
    run = Path(run_dir)
    eb = run / "EVIDENCE_BUNDLE.json"
    if not eb.exists():
        raise SealError("no EVIDENCE_BUNDLE.json: run is not sealed")
    bundle = json.loads(eb.read_text())
    now = collect(run)
    differ = sorted(r for r in set(now) | set(bundle["files"])
                    if now.get(r, {}).get("sha256") != bundle["files"].get(r, {}).get("sha256"))
    res = {"evidence_files_differ": differ, "table_matches_manifest": None, "manifest_sha256": None}
    mf = run / "MANIFEST.json"
    if mf.exists():
        man = json.loads(mf.read_text())
        t = Path(table) if table else run / RESULTS_DIR / man["table_path"]
        res["table_matches_manifest"] = (sha256_file(t) == man["table_sha256"]) if t.exists() else "table missing"
        res["manifest_sha256"] = sha256_file(mf)
    res["bundle_digest_matches"] = bundle_hash(now) == bundle["evidence_bundle_sha256"] == bundle_hash(bundle["files"])
    res["manifest_bundle_matches"] = not mf.exists() or man["evidence_bundle_sha256"] == bundle["evidence_bundle_sha256"]
    res["ok"] = not differ and res["bundle_digest_matches"] and res["manifest_bundle_matches"] and res["table_matches_manifest"] in (None, True)
    return res
