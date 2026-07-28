#!/usr/bin/env python3
"""Archive superseded coexistence conflicts and annotate new entries.

Resolves conflicts reported by
``vllm_hust_benchmark.integration._find_superseded_coexistence_conflicts``
by performing, for each conflict:

1. Moving the old submission directory into
   ``archive/suspect/<reason>-<YYYYMMDD>/<old_dirname>/``.
2. Writing a ``README.md`` inside the archived directory explaining the
   archival reason, the superseding entry, and the conflict signature.
3. Updating the new entry's ``run_leaderboard.json`` with
   ``metadata.supersedes`` (string when single, list when multiple)
   pointing to the old ``entry_id``, plus ``metadata.supersedes_reason``.

The script is **idempotent**: re-running on already-archived conflicts is
a no-op (it only ensures the new entry's ``supersedes`` annotation is in
place).

Usage::

    python scripts/archive_superseded_coexistence.py --dry-run
    python scripts/archive_superseded_coexistence.py --apply

Exit codes:
    0  success (or dry-run with pending actions)
    2  conflicts remain after --apply (unexpected; manual triage required)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = REPO_ROOT / "archive" / "suspect"
ARCHIVE_REASON = "superseded-coexistence-historical-pr-backfill"
SOURCE_DIR = REPO_ROOT / "submissions"

SUPERSEDES_REASON = (
    "historical-pr-backfill: same (engine_commit, plugin_commit) code "
    "combination as archived run; latest submitted_at wins per spec."
)


def _archive_dir_for_today() -> Path:
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    return ARCHIVE_ROOT / f"{ARCHIVE_REASON}-{date_tag}"


def _write_archive_readme(
    archived_dir: Path,
    *,
    old_entry_id: str,
    new_entry_id: str,
    new_dir_name: str,
    signature: str,
    archived_at: str,
) -> None:
    """Write README.md inside the archived dir. Never overwrites an existing one."""
    readme = archived_dir / "README.md"
    if readme.is_file():
        return
    readme.write_text(
        f"# Archived: superseded by `{new_entry_id}`\n"
        f"\n"
        f"- Archived at: {archived_at}\n"
        f"- Original path: `submissions/{archived_dir.name}/`\n"
        f"- Old entry_id: `{old_entry_id}`\n"
        f"- Superseded by: `{new_entry_id}` "
        f"(now at `submissions/{new_dir_name}/`)\n"
        f"- Reason: same `(engine_commit, plugin_commit)` code combination as the\n"
        f"  new entry. historical-pr-backfill data requires explicit `supersedes`\n"
        f"  annotation per `spec.md` § \"superseded 不得与新 OK 点共存\".\n"
        f"- Conflict signature: `{signature}`\n",
        encoding="utf-8",
    )


def _update_new_entry_supersedes(
    new_dir: Path,
    *,
    old_entry_id: str,
) -> bool:
    """Update new entry's metadata.supersedes to include ``old_entry_id``.

    Returns ``True`` if the file was modified, ``False`` if already up-to-date.
    Preserves an existing string-or-list shape; merges deduplicated.
    """
    artifact = new_dir / "run_leaderboard.json"
    if not artifact.is_file():
        return False

    payload_text = artifact.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        return False

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata

    existing = metadata.get("supersedes")
    if isinstance(existing, str):
        supersedes_list = [existing]
    elif isinstance(existing, list):
        supersedes_list = [str(x) for x in existing if x]
    else:
        supersedes_list = []

    if old_entry_id in supersedes_list:
        return False

    supersedes_list.append(old_entry_id)
    metadata["supersedes"] = (
        supersedes_list[0] if len(supersedes_list) == 1 else supersedes_list
    )
    metadata["supersedes_reason"] = SUPERSEDES_REASON

    new_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if new_text == payload_text:
        return False
    artifact.write_text(new_text, encoding="utf-8")
    return True


def _resolve_conflict(
    conflict: dict[str, Any],
    *,
    archive_root: Path,
    archived_at: str,
    apply: bool,
) -> dict[str, Any]:
    old_dir = Path(conflict["old_dir"])
    new_dir = Path(conflict["new_dir"])
    old_entry_id = conflict["old_entry_id"]
    new_entry_id = conflict["new_entry_id"]
    signature = conflict["signature"]

    if not old_dir.is_dir():
        return {"status": "skip_old_missing", "conflict": conflict}
    if not new_dir.is_dir():
        return {"status": "skip_new_missing", "conflict": conflict}

    archived_old = archive_root / old_dir.name
    if archived_old.is_dir():
        # Already archived; ensure the new entry annotation is in place too.
        annotated = False
        if apply:
            annotated = _update_new_entry_supersedes(
                new_dir, old_entry_id=old_entry_id
            )
        return {
            "status": "already_archived",
            "conflict": conflict,
            "archived_to": str(archived_old),
            "annotated_now": annotated,
        }

    action: dict[str, Any] = {
        "status": "pending",
        "conflict": conflict,
        "archived_to": str(archived_old),
    }
    if not apply:
        return action

    # 1. Move old dir into archive (atomic rename on same filesystem).
    archived_old.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_dir), str(archived_old))
    _write_archive_readme(
        archived_old,
        old_entry_id=old_entry_id,
        new_entry_id=new_entry_id,
        new_dir_name=new_dir.name,
        signature=signature,
        archived_at=archived_at,
    )
    # 2. Annotate new entry.
    _update_new_entry_supersedes(new_dir, old_entry_id=old_entry_id)

    action["status"] = "archived"
    return action


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without applying any changes.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply archival + supersedes annotation.",
    )
    args = parser.parse_args(argv)

    if not SOURCE_DIR.is_dir():
        print(f"ERROR: source dir not found: {SOURCE_DIR}", file=sys.stderr)
        return 2

    # Lazy import so --help works without the package on PYTHONPATH.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from vllm_hust_benchmark.integration import (
        _find_superseded_coexistence_conflicts,
    )

    conflicts = _find_superseded_coexistence_conflicts(SOURCE_DIR)
    if not conflicts:
        print("No superseded coexistence conflicts to resolve.")
        return 0

    archive_root = _archive_dir_for_today()
    archived_at = datetime.now(timezone.utc).isoformat()

    print(f"Resolving {len(conflicts)} conflict(s)...")
    print(f"Archive root: {archive_root}")
    print()

    actions: list[dict[str, Any]] = []
    for c in conflicts:
        action = _resolve_conflict(
            c,
            archive_root=archive_root,
            archived_at=archived_at,
            apply=args.apply,
        )
        actions.append(action)
        status = action["status"]
        old_name = Path(c["old_dir"]).name
        new_name = Path(c["new_dir"]).name
        if status == "pending":
            print(f"  WOULD ARCHIVE: {old_name}")
            print(f"    -> {action['archived_to']}")
            print(f"    AND annotate {new_name} with supersedes={c['old_entry_id']}")
        elif status == "archived":
            print(f"  ARCHIVED: {old_name} -> {action['archived_to']}")
            print(f"    annotated {new_name} with supersedes={c['old_entry_id']}")
        elif status == "already_archived":
            extra = " (re-annotated)" if action.get("annotated_now") else ""
            print(f"  ALREADY ARCHIVED: {old_name}{extra}")
        else:
            print(f"  SKIP ({status}): {old_name}")

    print()
    if args.apply:
        archived_count = sum(1 for a in actions if a["status"] == "archived")
        print(f"Applied {archived_count} archival(s).")

        # Re-verify: ensure no remaining conflicts.
        remaining = _find_superseded_coexistence_conflicts(SOURCE_DIR)
        if remaining:
            print(
                f"WARNING: {len(remaining)} conflict(s) still remain after apply:",
                file=sys.stderr,
            )
            for c in remaining:
                print(
                    f"  {Path(c['old_dir']).name} vs {Path(c['new_dir']).name}",
                    file=sys.stderr,
                )
            return 2
        print("Verification: 0 conflicts remaining.")
    else:
        pending = sum(1 for a in actions if a["status"] == "pending")
        print(f"Dry run: {pending} dir(s) would be archived.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
