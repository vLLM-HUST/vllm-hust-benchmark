"""Original JSONSchemaBench constraints with pinned publisher prompt and validator.

Local NPU API validation, not the reference OpenAIEngine or a performance run.
Original sources must be acquired separately; nothing is downloaded or rewritten.
"""

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import types
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

PROMPT_SHA = "27b9f83885c0982102c3aa973a4b00457d0335893db770d4c485ffe5eb08375e"  # pragma: allowlist secret - public source checksum
SCORER_SHA = "19623d44057e542e841a70c77e637cbfc58653960070eabc418704d62fb14526"  # pragma: allowlist secret - public source checksum
SCHEMAS = {
    "o10008.json": "9988edbc17dcf63d85a275f42d0a2ef6d63dec021580a9f4a3b6cf80ffd29b17",  # pragma: allowlist secret - public asset checksum
    "o10009.json": "6d1a5722bf3fade2fc2ef6dff0aa0ce3773d0d45fc7caca8ee5fe4b75e58f5f3",  # pragma: allowlist secret - public asset checksum
    "o10010.json": "1f31dec3794e24b6133fbb03a77876508940e3d866064881a9900cd8db84cbeb",  # pragma: allowlist secret - public asset checksum
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint(value):
    parsed = urlparse(value)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1"}
        and parsed.path in {"", "/"}
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment,
        "local HTTP origin without credentials required",
    )
    return value.rstrip("/")


def has_ref(value):
    if isinstance(value, dict):
        return "$ref" in value or any(has_ref(v) for v in value.values())
    return isinstance(value, list) and any(has_ref(v) for v in value)


def request_body(schema, messages, model):
    return {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": schema, "name": "json_schema"},
        },
        "stream": False,
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def post(origin, path, body):
    request = Request(
        origin + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-root",
        required=True,
        type=Path,
        help="directory containing Github_easy",
    )
    parser.add_argument(
        "--official-code",
        required=True,
        type=Path,
        help="original checkout containing core",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--origin", default="http://127.0.0.1:19466", type=endpoint)
    parser.add_argument("--context-capacity", required=True, type=int)
    args = parser.parse_args()
    os.umask(0o077)
    out = args.output.resolve()
    for root in (args.schema_root.resolve(), args.official_code.resolve()):
        require(
            not out.is_relative_to(root) and not root.is_relative_to(out),
            "output and original source trees must be separate",
        )
    core_path = args.official_code.resolve() / "core"
    require(
        digest(core_path / "messages.py") == PROMPT_SHA,
        "publisher prompt identity mismatch",
    )
    require(
        digest(core_path / "evaluator.py") == SCORER_SHA,
        "publisher validator identity mismatch",
    )
    for package, version in {"jsonschema": "4.23.0", "numpy": "2.2.3"}.items():
        require(metadata.version(package) == version, "pinned core dependency mismatch")
    assets = []
    for name, expected in SCHEMAS.items():
        path = args.schema_root / "Github_easy" / name
        require(digest(path) == expected, "original schema identity mismatch")
        schema = json.loads(path.read_text(encoding="utf-8"))
        require(
            isinstance(schema, (dict, bool)) and not has_ref(schema),
            "this profile requires self-contained original schemas",
        )
        assets.append((name, path, schema))
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib-private"))
    require("core" not in sys.modules, "core namespace already loaded")
    core = types.ModuleType("core")
    core.__path__ = [str(core_path)]
    sys.modules["core"] = core
    messages_module = importlib.import_module("core.messages")
    evaluator = importlib.import_module("core.evaluator")
    receipt = {
        "source_id": "JSONSCHEMABENCH",
        "status": "STARTED",
        "samples": [],
        "dataset_revision": "9a94995b9279ae3af3aed4b2629172790b968d14",  # pragma: allowlist secret - public revision
        "code_revision": "8003e8405c4d8d8b327b1eb472c9856297d75493",  # pragma: allowlist secret - public revision
        "schema_adaptation": False,
        "optional_engine_initializer_skipped": True,
        "full_reference_environment_reproduced": False,
        "performance_claim": False,
        "full_suite_claim": False,
        "model": args.model,
        "profile": "original schema, publisher two-shot prompt and validator, local nonstreaming API",
        "transport_deviations": [
            "local model",
            "nonstreaming",
            "temperature=0",
            "max_tokens=1024",
            "enable_thinking=false",
        ],
    }
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for name, path, schema in assets:
        sample = {"file": name, "schema_sha256": SCHEMAS[name]}
        try:
            require(
                evaluator.is_json_schema_valid(schema),
                "publisher rejects original schema",
            )
            messages = messages_module.few_shots_messages_formatter(
                "Github_easy", schema
            )
            body = request_body(schema, messages, args.model)
            count = post(
                args.origin,
                "/tokenize",
                {
                    "model": args.model,
                    "messages": messages,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )["count"]
            require(
                type(count) is int and count + 1024 <= args.context_capacity,
                "full prompt exceeds declared context capacity",
            )
            response = post(args.origin, "/v1/chat/completions", body)
            (out / (name + ".private.json")).write_text(
                json.dumps({"request": body, "response": response}), encoding="utf-8"
            )
            choice = response["choices"][0]
            try:
                value = json.loads(choice["message"]["content"])
                parse_valid = True
                conforms = bool(evaluator.validate_json_schema(value, schema))
            except (ValueError, TypeError):
                parse_valid = conforms = False
            sample.update(
                status="GENERATED",
                finish_reason=choice["finish_reason"],
                usage=response["usage"],
                json_parse_valid=parse_valid,
                original_validator_valid=conforms,
                full_prompt_token_count_matches=response["usage"]["prompt_tokens"]
                == count,
            )
        except HTTPError as error:
            (out / (name + ".error.private.json")).write_bytes(error.read())
            sample.update(status="REQUEST_REJECTED", http_status=error.code)
        except Exception as error:
            sample.update(status="EXECUTION_FAILED", error_type=type(error).__name__)
        require(digest(path) == SCHEMAS[name], "original schema bytes changed")
        require(
            schema == json.loads(path.read_text(encoding="utf-8")),
            "original schema object changed",
        )
        receipt["samples"].append(sample)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt["status"] = "ORIGINAL_SCHEMA_PROFILE_ATTEMPTED"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
