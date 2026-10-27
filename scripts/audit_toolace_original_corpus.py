"""Portable original ToolACE preparation audit, not an online Agent benchmark.

Reuses existing source adapter/index/resolve. No model, tool execution or scoring.
"""

import argparse
from collections import Counter
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import zipfile

from prepare_xsum_original_assets import fetch, sha, verify

REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"  # pragma: allowlist secret - original source revision
DATA_SHA = "ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e"  # pragma: allowlist secret - original source checksum
CARD_SHA = "4e82e47712cfdc0bc86b3b0c6b03f5f2f00a1ca54ccfa696276833743c85e460"  # pragma: allowlist secret - original publisher documentation checksum


def installed_wheel_audit(wheel):
    import vllm_hust_workload_data as provider

    package = "vllm_hust_workload_data"
    distribution = metadata.distribution("vllm-hust-workload-data")
    wheel_metadata = distribution.read_text("WHEEL")
    if not wheel_metadata or "Wheel-Version:" not in wheel_metadata:
        raise ValueError("source/egg metadata is not installed-wheel proof")
    root = Path(distribution.locate_file(package)).resolve()
    if Path(provider.__file__).resolve() != root / "__init__.py":
        raise ValueError(
            "installed wheel import required; editable/source import is not installation proof"
        )
    direct = distribution.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise ValueError("editable import is not pinned installed-wheel proof")
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("duplicate wheel members")
        for name in archive.namelist():
            if not name.startswith(package + "/") or name.endswith("/"):
                continue
            relative = PurePosixPath(name)
            target = Path(distribution.locate_file(name)).resolve()
            if (
                ".." in relative.parts
                or "\\" in name
                or not target.is_relative_to(root)
            ):
                raise ValueError("wheel payload escaped provider package")
            if target.read_bytes() != archive.read(name):
                raise ValueError("installed provider differs from exact wheel")
            count += 1
    if not count:
        raise ValueError("provider wheel payload is empty")
    return {
        "status": "EXACT_INSTALLED_PROVIDER_PAYLOAD_MATCHED",
        "payloads": count,
        "wheel_sha256": sha(wheel),
    }


def audit(
    data_root, output, *, fetch_missing_card=False, wheel=None, catalog_root=None
):
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import run, prepared_rows
    from vllm_hust_workload_data.registry import source_spec

    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError(
            "fresh private audit output separate from original data required"
        )
    source = source_spec("TOOLACE", catalog_root)
    files = {item["path"]: item for item in source["files"]}
    if (
        source.get("revision") != REVISION
        or set(files) != {"README.md", "data.json"}
        or files["data.json"]["sha256"] != DATA_SHA
        or files["README.md"]["sha256"] != CARD_SHA
    ):
        raise ValueError("exact reviewed original ToolACE corpus profile required")
    data, card = data_root / "data.json", data_root / "README.md"
    if data.is_symlink() or card.is_symlink():
        raise ValueError("explicit nonsymlink original assets required")
    verify(data, files["data.json"])
    output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": "TOOLACE",
        "source_revision": REVISION,
        "status": "STARTED",
        "model_calls": 0,
        "tool_calls_executed": 0,
        "native_task_scorer_executed": False,
        "source_checkbox_eligible": False,
        "online_agent_environment": False,
        "request_renderer_verified": False,
        "raw_payloads_published": False,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        if not card.exists():
            if not fetch_missing_card:
                raise FileNotFoundError(
                    "publisher card absent; explicit --fetch-missing-card required"
                )
            if card.with_name(card.name + ".partial").exists():
                raise ValueError(
                    "previous partial publisher card retained; do not overwrite"
                )
            fetch(card, files["README.md"], files["README.md"]["url"])
        verify(card, files["README.md"])
        receipt["publisher_card_verified"] = True
        receipt["provider_artifact"] = (
            installed_wheel_audit(wheel)
            if wheel
            else {"status": "CURRENT_IMPORT_ONLY_NO_INSTALLED_WHEEL_PROOF"}
        )
        rows = json.loads(data.read_text("utf-8"))
        if not isinstance(rows, list) or len(rows) != files["data.json"]["rows"]:
            raise ValueError("full original corpus row count required")
        selected = sorted(
            {*range(min(3, len(rows))), *range(max(0, len(rows) - 3), len(rows))}
        )
        records, roles, layouts = {}, Counter(), Counter()
        standard_roles = {
            "human",
            "gpt",
            "user",
            "assistant",
            "system",
            "tool",
            "function",
            "observation",
        }
        count = 0
        for index, (original, record) in enumerate(
            zip(
                rows,
                iter_workload(
                    "TOOLACE",
                    data_root=data_root,
                    files=["data.json"],
                    catalog_root=catalog_root,
                ),
                strict=True,
            )
        ):
            if (
                set(original) != {"system", "conversations"}
                or not isinstance(original["system"], str)
                or record.task["input"] != original
                or record.outcome_record() is not None
                or record.runtime["kind"] != "historical_dialogue"
            ):
                raise ValueError(
                    "complete original system/dialogue and historical semantics required"
                )
            for turn in original["conversations"]:
                if not isinstance(turn, dict):
                    raise ValueError("original dialogue message object required")
                layout = (
                    "from_value"
                    if set(turn) == {"from", "value"}
                    else (
                        "role_content"
                        if set(turn) == {"role", "content"}
                        else "OTHER_UNRENDERED_LAYOUT"
                    )
                )
                layouts[layout] += 1
                role = turn.get("from", turn.get("role"))
                roles[
                    role
                    if isinstance(role, str) and role in standard_roles
                    else "OTHER_UNRENDERED_ROLE"
                ] += 1
            try:
                record.request_body()
            except ValueError:
                pass
            else:
                raise ValueError(
                    "historical dialogue must not become an original API capture"
                )
            if index in selected:
                records[index] = record
            count += 1
        binding = bind_source(
            "TOOLACE",
            data_root=data_root,
            files=["data.json"],
            revision=source["revision"],
            catalog_root=catalog_root,
        )
        manifest = output / "binding.private.json"
        manifest.write_text(json.dumps(binding), encoding="utf-8")
        run(
            {
                "sources": [
                    {
                        "source_id": "TOOLACE",
                        "data_root": str(data_root),
                        "manifest": str(manifest),
                    }
                ]
            },
            output=output / "prepared",
            catalog_root=catalog_root,
        )
        refs = list(prepared_rows(output / "prepared/workload.sqlite"))
        if len(refs) != count:
            raise ValueError("prepared original row count mismatch")
        for index in selected:
            resolved = resolve_input(
                output / "prepared/workload.sqlite",
                refs[index]["record_id"],
                data_root=data_root,
            )
            if (
                resolved.task != records[index].task
                or resolved.outcome_record() is not None
            ):
                raise ValueError("original prepared dialogue roundtrip mismatch")
        verify(data, files["data.json"])
        verify(card, files["README.md"])
        receipt.update(
            status="ORIGINAL_CORPUS_PREPARED_NOT_MODEL_OR_AGENT_EXECUTION",
            original_rows_equal=count,
            prepared_rows=len(refs),
            selected_original_indices=selected,
            original_input_roundtrips=len(selected),
            message_layout_counts=dict(layouts),
            known_speaker_contract_counts=dict(roles),
            contract_counts_are_format_validation_not_research_analysis=True,
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-wheel", type=Path)
    parser.add_argument("--fetch-missing-card", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.data_root,
        args.output,
        fetch_missing_card=args.fetch_missing_card,
        wheel=args.provider_wheel,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "original_rows_equal",
                    "original_input_roundtrips",
                    "model_calls",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
