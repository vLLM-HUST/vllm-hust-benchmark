# Public shared-adapter 0.9.0 artifact audit

PyPI now provides `vllm-ascend-kvcompress-hust==0.9.0`, uploaded on 2026-10-08. This removes literal
public-download unavailability as the reason to defer its inspection. The downloaded wheel matches
the registry's SHA-256, recorded in `receipt.json`. The PyramidKV PyPI project endpoint returned
HTTP 404 at the observation time; this does not describe other package indexes.

**The public shared wheel is not the source revision qualified by the current Qwen3.5 runbook.**
Both declare version 0.9.0, so version equality cannot transfer the qualification:

- The published wheel carries Manifest 0.2; the qualified source carries Manifest 0.3.
- Six Python files differ from qualified commit `19f322130e1b3841da953b1b421bcf3259e9b8dc`.
  `wheel-vs-qualified.json` records the SHA-256 of every wheel Python file and its source
  counterpart.
- Manual review of the provider diff confirms the published synthetic FULL-capture branch still
  returns the original metadata without binding the per-layer slot buffers. The qualified source
  binds those buffers for capture and clears padded replay slots. This is the graph-cache-write fix
  tracked by [shared PR #19](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/19),
  whose absence previously caused incorrect decoding in the recorded source evaluation. The
  published provider also has other differences in hybrid page sizes and cache layout checks; it is
  not a one-line substitute for the qualified source.

This audit reads the archive; it does not install or execute the public wheel and does not claim a
new runtime failure from that wheel. The old failure and corrected source-run evidence remain in
[the paired evaluation receipt](../2026-10-09-paired-evaluation/README.md). Continue using the exact
source pin in [installation instructions](../../../docs/install-and-rollback.md) for this host. Do
not replace it with unqualified `==0.9.0` solely because the version matches.

## Reproduce the static comparison

Download the exact wheel URL in `receipt.json`, verify its SHA-256, and use a checkout of the shared
repository containing the qualified commit. Python's standard library and Git are sufficient:

```python
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

receipt = json.loads(Path("receipt.json").read_text())
wheel = Path(receipt["wheel"]["filename"])
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == receipt["wheel"]["digests"]["sha256"]
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if not name.endswith(".py"):
            continue
        source = subprocess.check_output(
            [
                "git",
                "-C",
                "/path/to/shared-repository",
                "show",
                receipt["qualified_source_revision"] + ":src/" + name,
            ]
        )
        print(name, hashlib.sha256(archive.read(name)).hexdigest(), hashlib.sha256(source).hexdigest())
```

`provider-wheel-vs-qualified.diff` preserves the reviewed code differences; its source attribution
and licensing remain those of the linked Apache-2.0 shared repository. No wheel or third-party
model/data assets are republished here. `SHA256SUMS` binds this report and the comparison files.

## Remaining release gate

The shared package release owner must select and publish a corrected, immutable artifact/version
through the project's approved release channel. The source backport validated on this older host and
the newer main-branch PR are distinct compatibility targets; a release must declare which one it
supports. Then repeat isolated installation, Manager composition, graph/MTP/APC serving, quality
checks and rollback against the exact published artifact. Local source validation cannot replace
that acceptance, and availability on PyPI alone does not satisfy it. This gate does not block
continued source-based performance or capacity work in issue #1.
