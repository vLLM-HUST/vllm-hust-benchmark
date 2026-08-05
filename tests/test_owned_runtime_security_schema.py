import json
from pathlib import Path

from jsonschema import Draft7Validator


SCHEMA_PATH = (
    Path(__file__).parent.parent / "schemas" / "owned_runtime_security_v1.schema.json"
)


def _validator() -> Draft7Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


def test_owned_runtime_security_schema_accepts_both_explicit_modes() -> None:
    validator = _validator()
    assert not list(
        validator.iter_errors(
            {
                "schema_version": "owned-runtime-security/v1",
                "privileged": False,
                "authorization_source": None,
            }
        )
    )
    assert not list(
        validator.iter_errors(
            {
                "schema_version": "owned-runtime-security/v1",
                "privileged": True,
                "authorization_source": "user-explicit:thread-019fc873:2026-08-05",
            }
        )
    )


def test_owned_runtime_security_schema_rejects_unbound_authorization() -> None:
    validator = _validator()
    invalid = [
        {
            "schema_version": "owned-runtime-security/v1",
            "privileged": True,
            "authorization_source": None,
        },
        {
            "schema_version": "owned-runtime-security/v1",
            "privileged": False,
            "authorization_source": "user-explicit:thread-019fc873",
        },
        {
            "schema_version": "owned-runtime-security/v1",
            "privileged": True,
            "authorization_source": "$(unsafe)",
        },
    ]
    assert all(list(validator.iter_errors(payload)) for payload in invalid)
