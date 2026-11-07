"""Prepare a separately labelled Qwen3.5 template for original SoM messages."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
OLD = """        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}"""
NEW = """        {%- if not loop.first %}
            {%- if message.name is not defined or message.name not in ['example_user', 'example_assistant'] %}
                {{- raise_exception('System message must be at the beginning.') }}
            {%- endif %}
            {{- '<|im_start|>system\\n[' + message.name + ']\\n' + content + '<|im_end|>\\n' }}
        {%- endif %}"""


def prepare(template_path, output):
    raw = template_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            "Unverified Qwen3.5 template; expected the audited source SHA256"
        )
    source = raw.decode("utf-8")
    if source.count(OLD) != 1:
        raise ValueError("Expected exactly one original subsequent-system guard")
    patched = source.replace(OLD, NEW, 1).encode("utf-8")
    with output.open("xb") as stream:
        stream.write(patched)
    return {
        "profile": "QWEN35_LOCAL_SOM_NAMED_SYSTEM_COMPATIBILITY",
        "source_template_sha256": SOURCE_SHA256,
        "output_template_sha256": hashlib.sha256(patched).hexdigest(),
        "original_request_schema_changed": False,
        "native_model_or_browser_executed": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
