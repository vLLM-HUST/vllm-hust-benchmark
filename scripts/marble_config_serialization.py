"""Lossless JSON-as-YAML for original configurations, including control chars."""

import json
import re

# JSON already escapes U+0000..001F. YAML1.1 additionally rejects C1/DEL and
# normalizes literal NEL/LS/PS. Escape those codepoints without converting astral
# characters to surrogate pairs (YAML and JSON decode such pairs differently).
YAML_ESCAPES = re.compile(r"[\x7f-\x9f\u2028\u2029\ud800-\udfff\ufffe\uffff]")


def native_config_json(value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    return YAML_ESCAPES.sub(lambda match: f"\\u{ord(match.group()):04x}", raw)
