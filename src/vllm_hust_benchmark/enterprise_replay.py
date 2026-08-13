from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "enterprise-dataset-registry/v1"
AUTHORIZATION_SCHEMA_VERSION = "enterprise-data-authorization/v1"
DATA_ROOT_ENV = "VLLM_HUST_ENTERPRISE_DATA_ROOT"
AUTHORIZATION_FILE_ENV = "VLLM_HUST_ENTERPRISE_AUTHORIZATION_FILE"
DEFAULT_AUTHORIZATION_FILENAME = ".vllm-hust-enterprise-authorized.json"
SAMPLER_VERSION = "enterprise-replay-sampler/v1"


class EnterpriseReplayError(RuntimeError):
    """Raised when enterprise replay cannot proceed without weakening a gate."""


@dataclass(frozen=True)
class EnterpriseReplayRequest:
    case_id: str
    dataset_id: str
    request_id: str
    source_index: int
    source_model_name: str
    request_type: str
    request_body: dict[str, Any]
    workload_family_id: str

    def to_openai_payload(self, *, served_model: str) -> dict[str, Any]:
        payload = dict(self.request_body)
        payload["model"] = served_model
        return payload

    def redacted_record(self, *, served_model: str) -> dict[str, Any]:
        payload = self.to_openai_payload(served_model=served_model)
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        prompt_text = _request_prompt_text(payload)
        return {
            "schema_version": "enterprise-replay-redacted/v1",
            "case_id": self.case_id,
            "dataset_id": self.dataset_id,
            "request_id": self.request_id,
            "source_index": self.source_index,
            "source_model_name": self.source_model_name,
            "served_model": served_model,
            "request_type": self.request_type,
            "workload_family_id": self.workload_family_id,
            "request_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt_text),
            "prompt_tokens_est": _fallback_token_count(prompt_text),
            "max_output_tokens": _request_output_len(payload),
            "message_count": len(payload.get("messages") or []),
            "stream": bool(payload.get("stream", False)),
            "tool_count": len(payload.get("tools") or []),
            "has_response_format": isinstance(payload.get("response_format"), dict),
        }


def _registry_resource() -> Any:
    return resources.files("vllm_hust_benchmark.data").joinpath(
        "enterprise_dataset_registry.json"
    )


def validate_enterprise_dataset_registry(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"enterprise registry schema_version must be {SCHEMA_VERSION!r}")
    datasets = payload.get("datasets")
    cases = payload.get("cases")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("enterprise registry datasets must be a non-empty array")
    if not isinstance(cases, list) or not cases:
        raise ValueError("enterprise registry cases must be a non-empty array")

    dataset_ids: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("enterprise registry dataset entries must be objects")
        dataset_id = str(dataset.get("dataset_id") or "").strip()
        relative_path = str(dataset.get("relative_path") or "").strip()
        if not dataset_id or dataset_id in dataset_ids:
            raise ValueError(f"invalid or duplicate enterprise dataset_id: {dataset_id!r}")
        if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"{dataset_id}: relative_path must stay below the explicit data root")
        if "__MACOSX" in Path(relative_path).parts or Path(relative_path).name.startswith("._"):
            raise ValueError(f"{dataset_id}: macOS metadata files are not replay assets")
        if dataset.get("source_format") not in {
            "wrapped_openai_request_jsonl",
            "direct_messages_jsonl",
        }:
            raise ValueError(f"{dataset_id}: unsupported source_format")
        if int(dataset.get("record_count") or 0) < 1:
            raise ValueError(f"{dataset_id}: record_count must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", str(dataset.get("sha256") or "")):
            raise ValueError(f"{dataset_id}: sha256 must be lowercase hex")
        dataset_ids.add(dataset_id)

    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("enterprise registry case entries must be objects")
        case_id = str(case.get("case_id") or "").strip()
        if not case_id or case_id in case_ids:
            raise ValueError(f"invalid or duplicate enterprise case_id: {case_id!r}")
        if case.get("dataset_id") not in dataset_ids:
            raise ValueError(f"{case_id}: unknown dataset_id {case.get('dataset_id')!r}")
        case_ids.add(case_id)


@lru_cache(maxsize=1)
def load_enterprise_dataset_registry() -> dict[str, Any]:
    with _registry_resource().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("enterprise registry must be a JSON object")
    validate_enterprise_dataset_registry(payload)
    return payload


def enterprise_dataset_rows() -> list[dict[str, Any]]:
    return [dict(row) for row in load_enterprise_dataset_registry()["datasets"]]


def enterprise_case_rows() -> list[dict[str, Any]]:
    return [dict(row) for row in load_enterprise_dataset_registry()["cases"]]


def resolve_enterprise_data_root(root: str | os.PathLike[str] | None = None) -> Path:
    raw = str(root) if root is not None else os.environ.get(DATA_ROOT_ENV)
    if not raw:
        raise EnterpriseReplayError(
            f"enterprise data root is required; pass --data-root or set {DATA_ROOT_ENV}"
        )
    resolved = Path(raw).expanduser().resolve()
    if not resolved.is_dir():
        raise EnterpriseReplayError(f"enterprise data root is not a directory: {resolved}")
    return resolved


def _resolve_authorization_file(
    data_root: Path, authorization_file: str | os.PathLike[str] | None
) -> Path:
    raw = (
        str(authorization_file)
        if authorization_file is not None
        else os.environ.get(AUTHORIZATION_FILE_ENV)
    )
    return Path(raw).expanduser().resolve() if raw else data_root / DEFAULT_AUTHORIZATION_FILENAME


def _require_dataset_authorization(
    *,
    dataset_id: str,
    data_root: Path,
    authorization_file: str | os.PathLike[str] | None,
) -> None:
    path = _resolve_authorization_file(data_root, authorization_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnterpriseReplayError(
            f"valid enterprise authorization file is required: {path}"
        ) from exc
    if payload.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise EnterpriseReplayError("enterprise authorization schema version mismatch")
    authorized = payload.get("authorized_dataset_ids")
    if not isinstance(authorized, list) or dataset_id not in authorized:
        raise EnterpriseReplayError(f"dataset is not authorized for replay: {dataset_id}")


def _find_registry_row(
    registry: Mapping[str, Any], *, key: str, value: str, section: str
) -> dict[str, Any]:
    for row in registry[section]:
        if row.get(key) == value:
            return dict(row)
    raise EnterpriseReplayError(f"unknown enterprise {key}: {value}")


def _asset_path(data_root: Path, relative_path: str) -> Path:
    candidate = (data_root / relative_path).resolve()
    if not candidate.is_relative_to(data_root):
        raise EnterpriseReplayError("enterprise asset escaped the explicit data root")
    return candidate


def verify_enterprise_dataset_asset(dataset: Mapping[str, Any], *, data_root: Path) -> Path:
    path = _asset_path(data_root, str(dataset["relative_path"]))
    if not path.is_file():
        raise EnterpriseReplayError(f"enterprise dataset asset is missing: {path}")
    digest = hashlib.sha256()
    record_count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                record_count += 1
    if digest.hexdigest() != dataset["sha256"]:
        raise EnterpriseReplayError(f"enterprise dataset checksum mismatch: {dataset['dataset_id']}")
    if record_count != int(dataset["record_count"]):
        raise EnterpriseReplayError(
            f"enterprise dataset record count mismatch: expected {dataset['record_count']}, got {record_count}"
        )
    return path


def _render_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return str(content.get("text") or "")
    if isinstance(content, list):
        return "\n".join(filter(None, (_render_message_content(item) for item in content)))
    return str(content or "")


def _request_prompt_text(request_body: Mapping[str, Any]) -> str:
    messages = request_body.get("messages") or []
    if messages:
        return "\n\n".join(
            f"{str(message.get('role') or '').strip()}: {_render_message_content(message.get('content'))}".strip()
            for message in messages
            if isinstance(message, Mapping)
        )
    return str(request_body.get("prompt") or "")


def _fallback_token_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", text, re.UNICODE))


def _request_output_len(request_body: Mapping[str, Any]) -> int:
    raw = request_body.get("max_tokens", request_body.get("max_completion_tokens", 0))
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _validate_request_body(request_body: Mapping[str, Any]) -> None:
    messages = request_body.get("messages")
    prompt = request_body.get("prompt")
    if not isinstance(messages, list) and not isinstance(prompt, str):
        raise EnterpriseReplayError("request must contain messages or prompt")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or not str(message.get("role") or "").strip():
                raise EnterpriseReplayError("request messages must contain a role")
            content = message.get("content")
            if not isinstance(content, (str, list, dict)):
                raise EnterpriseReplayError("request message content has an unsupported shape")
    if "stream" in request_body and not isinstance(request_body["stream"], bool):
        raise EnterpriseReplayError("request stream field must be boolean")
    if "tools" in request_body and not isinstance(request_body["tools"], list):
        raise EnterpriseReplayError("request tools field must be an array")
    if "response_format" in request_body and not isinstance(request_body["response_format"], dict):
        raise EnterpriseReplayError("request response_format field must be an object")


def _parse_source_row(
    row: Mapping[str, Any], *, source_format: str, fallback_source_model: str
) -> tuple[str, str, dict[str, Any]]:
    if source_format == "wrapped_openai_request_jsonl":
        raw_body = row.get("request_body")
        if isinstance(raw_body, str):
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise EnterpriseReplayError("wrapped request_body is not valid JSON") from exc
        elif isinstance(raw_body, dict):
            body = dict(raw_body)
        else:
            raise EnterpriseReplayError("wrapped request_body must be an object or JSON string")
        source_model = str(row.get("model_name") or body.get("model") or fallback_source_model)
        request_type = str(row.get("request_type") or "ChatCompletion")
    else:
        allowed = {
            "messages", "prompt", "stream", "tools", "tool_choice", "response_format",
            "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed", "n"
        }
        body = {key: row[key] for key in allowed if key in row}
        if not isinstance(body.get("messages"), list) and isinstance(row.get("prompt"), str):
            body["messages"] = [{"role": "user", "content": row["prompt"]}]
        source_model = str(row.get("model_name") or row.get("model") or fallback_source_model)
        request_type = str(row.get("request_type") or "ChatCompletion")
    if not isinstance(body, dict):
        raise EnterpriseReplayError("request body must decode to an object")
    _validate_request_body(body)
    return source_model, request_type, body


def _combined_request_text(request_body: Mapping[str, Any]) -> str:
    return _request_prompt_text(request_body).lower()


def _matches_filter(filter_id: str | None, request_body: Mapping[str, Any]) -> bool:
    if filter_id is None:
        return True
    messages = request_body.get("messages") or []
    text = _combined_request_text(request_body)
    if filter_id == "normal_chat_support_multi_turn":
        return len(messages) >= 4 and "user simulation guidelines" in text and "support representative" in text
    if filter_id == "normal_chat_single_turn_general":
        excluded = (
            "act as an impartial judge", "reference answer", "output only one valid json object",
            "expert document analyzer", "collection order analysis expert", "trajectory-reading assistant"
        )
        return len(messages) == 1 and not any(phrase in text for phrase in excluded)
    if filter_id == "json_structured_extraction":
        return any(
            phrase in text
            for phrase in (
                "extract structured", "extract information", "analyze the chat history",
                "named entity recognition", "relation extraction", "canonical product search query"
            )
        )
    if filter_id == "json_planning_orchestration":
        return any(
            phrase in text
            for phrase in (
                "orchestration planner", "tool schemas", "tool-powered analysis",
                "return only valid json that defines depth iteration goals"
            )
        )
    if filter_id == "code_reward_hack_judge":
        return "strict code-integrity reviewer" in text or "reward hack" in text
    if filter_id == "code_answer_correctness_judge":
        return "judging answer correctness" in text or "answer correctness" in text
    raise EnterpriseReplayError(f"unknown enterprise replay filter: {filter_id}")


def _sample_score(*, dataset_id: str, case_id: str, seed: int, source_index: int) -> int:
    material = f"{SAMPLER_VERSION}\0{dataset_id}\0{case_id}\0{seed}\0{source_index}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest(), 16)


def load_enterprise_replay_requests(
    case_id: str,
    *,
    data_root: str | os.PathLike[str] | None,
    authorization_file: str | os.PathLike[str] | None = None,
    limit: int = 100,
    seed: int = 0,
    registry: Mapping[str, Any] | None = None,
) -> list[EnterpriseReplayRequest]:
    if limit < 1:
        raise ValueError("enterprise replay limit must be positive")
    active_registry = dict(registry or load_enterprise_dataset_registry())
    validate_enterprise_dataset_registry(active_registry)
    case = _find_registry_row(active_registry, key="case_id", value=case_id, section="cases")
    dataset = _find_registry_row(
        active_registry, key="dataset_id", value=case["dataset_id"], section="datasets"
    )
    root = resolve_enterprise_data_root(data_root)
    _require_dataset_authorization(
        dataset_id=dataset["dataset_id"],
        data_root=root,
        authorization_file=authorization_file,
    )
    path = verify_enterprise_dataset_asset(dataset, data_root=root)

    heap: list[tuple[int, int, EnterpriseReplayRequest]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EnterpriseReplayError(
                    f"invalid JSONL record at source index {source_index}"
                ) from exc
            if not isinstance(row, dict):
                raise EnterpriseReplayError("enterprise JSONL records must be objects")
            source_model, request_type, request_body = _parse_source_row(
                row,
                source_format=dataset["source_format"],
                fallback_source_model=dataset["source_model_name"],
            )
            if not _matches_filter(case.get("filter_id"), request_body):
                continue
            raw_request_id = row.get("request_id") or row.get("id") or source_index
            source_identity = f"{dataset['dataset_id']}\0{raw_request_id}\0{source_index}"
            request_id = (
                f"{dataset['dataset_id']}::"
                f"{hashlib.sha256(source_identity.encode('utf-8')).hexdigest()[:24]}"
            )
            request = EnterpriseReplayRequest(
                case_id=case_id,
                dataset_id=dataset["dataset_id"],
                request_id=request_id,
                source_index=source_index,
                source_model_name=source_model,
                request_type=request_type,
                request_body=request_body,
                workload_family_id=dataset["workload_family_id"],
            )
            score = _sample_score(
                dataset_id=dataset["dataset_id"],
                case_id=case_id,
                seed=seed,
                source_index=source_index,
            )
            item = (-score, -source_index, request)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = [(-neg_score, -neg_index, request) for neg_score, neg_index, request in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return [request for _, _, request in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and sample enterprise replay data")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--data-root")
    parser.add_argument("--authorization-file")
    parser.add_argument("--served-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.list_cases:
        payload: Any = enterprise_case_rows()
    else:
        if not args.case_id:
            parser.error("--case-id is required unless --list-cases is used")
        requests = load_enterprise_replay_requests(
            args.case_id,
            data_root=args.data_root,
            authorization_file=args.authorization_file,
            limit=args.limit,
            seed=args.seed,
        )
        payload = [
            request.redacted_record(served_model=args.served_model) for request in requests
        ]
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
