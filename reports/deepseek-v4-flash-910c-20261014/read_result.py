"""Reassemble a complete native benchmark result from lossless request shards."""

import gzip
import hashlib
import json
import sys
from pathlib import Path


def read_result(index_path):
    path = Path(index_path)
    index = json.loads(path.read_text())
    result = json.loads(gzip.decompress((path.parent / index["metadata"]).read_bytes()))
    for key in index["request_fields"]:
        result[key] = []
    for name in index["request_chunks"]:
        chunk = json.loads(gzip.decompress((path.parent / name).read_bytes()))
        for key in index["request_fields"]:
            result[key].extend(chunk[key])
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == index["semantic_json_sha256"]
    return result


if __name__ == "__main__":
    json.dump(read_result(sys.argv[1]), sys.stdout, ensure_ascii=False)
    print()
