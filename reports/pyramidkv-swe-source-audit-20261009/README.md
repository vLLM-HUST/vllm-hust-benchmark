# SWE-bench-Pro V2 execution-source audit

This advances [PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8) and
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). All 642 tasks in the
pinned V2 default split now have verified task, repository/base-commit, verifier, image and
root-license metadata. **No image layers were downloaded, no sandbox tasks ran, and there is no
resolved-rate or optimization result.** The runtime and task-rights gates remain open.

Sources are the HF `ScaleAI/SWE-bench_Pro` revision `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50` and
upstream
[Harbor-format harness](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2)
revision `66f92766bba642462d4bbe5479e83f91f9211862`. The Parquet SHA-256 is recorded in
`summary.json`. The 642 task IDs are unique and cover eleven repositories; no hard-only or V1
substitution was made.

## Verified mappings

- Every task has the required task descriptor, instruction, Dockerfile, test configuration, test
  entrypoint, run script, parser, test patch and reference patch in the pinned Git tree. Downloaded
  descriptors/configuration bytes were checked against their Git blob IDs.
- Dataset test lists match the harness lists. Some harness fields still use Python list literals;
  these are parsed with `ast.literal_eval` and accepted only as lists of strings. Dataset fields
  remain strict JSON. No task-provided commands are executed by the auditor.
- HF image tags match `task.toml` and the single `FROM` line in each Dockerfile. Registry manifests,
  index members and config blobs were checked against their SHA-256 digests. All selected images are
  Linux/AMD64. Both the tag-resolved digest and platform manifest digest are retained.
- Each task's root-license file was retrieved at that exact repository base commit and checked
  against its Git blob ID and SHA-256. This identifies root-license provenance, not blanket rights
  clearance for task content or dependencies. GitHub classified 417 as GPL-3.0, 115 as Apache-2.0,
  105 as AGPL-3.0 and five as NOASSERTION. The latter five share one Element Web license file;
  NOASSERTION means the API did not assign an SPDX identifier, not that the file is absent.

All 642 reference patches match the harness byte-for-byte. **One test patch differs:** the Ansible
instance listed in `test-patch-byte-difference.json` has 166 CRLF sequences in its multipart fixture
in the harness, while the HF field has LF. The remaining 641 test patches match exactly. Fixture
line endings can affect tests: the verifier authority is the pinned harness's original bytes, and
the normalized HF text must not be substituted. Both hashes are retained. Any other unexplained
patch change fails the audit. Actual fresh-sandbox verification is still required.

## Resource and execution limits

The unique compressed layers total 544,122,719,405 bytes (about 506.8 GiB); the largest individual
image's compressed layers total 6,518,064,990 bytes. These are registry metadata sizes, not measured
unpacked storage or HBM. A full pre-pull would exceed this pod's available disk. Sequential pulling
and reclamation could reduce storage needs, so full-cache capacity is not a universal requirement.

This pod is ARM64 and lacks an available container backend; the mount/network namespace probe
failed. The frozen images select AMD64. Execution needs a supported isolated AMD64 backend or an
explicitly validated compatible environment, plus access from that backend to the model endpoint. A
Docker CLI alone does not establish those capabilities. Agent/scaffold/tool budgets, task rights,
and fresh-sandbox regrading still need their execution contract. Image metadata does not prove that
the image's checked-out repository HEAD or tests match the declared task; those checks must run
inside the sandbox.

## Evidence and reproduction

`task-inventory.part-*.jsonl.gz` concatenates in numeric order to the complete inventory;
`audit-receipt.json` records its uncompressed SHA-256. Separate compressed retrieval receipts retain
the initial unresolved-index records and subsequent immutable platform resolutions. No source
questions, patches, license text or image-config environment values are republished. Two offline
audit generations were byte-identical; all 642 passed, with no unresolved audit failures.

Recreate the local metadata input directory from the pinned sources: retrieve the recursive harness
Git tree, each task's three descriptor/configuration files, root license at `base_commit`, and
registry manifests/config blobs by the published digests. Public GitHub API authentication may be
needed for rate limits; registry read tokens are never part of the evidence. Retain the original
upstream test patch for the documented CRLF case. The directory layout is checked by the auditor and
its focused fixture tests. Then run:

```bash
python scripts/audit_pyramidkv_swe.py \
  --data /path/to/pinned-default-test.parquet \
  --tree /path/to/swe-harness-tree.json \
  --assets /path/to/metadata-input-directory \
  --output /path/to/new-empty-audit
```

Python 3.11+ and PyArrow are required. This command verifies all 642 source rows and reports
failures without dropping tasks. Missing source metadata or a changed image/config digest cannot
produce a successful audit. It does not launch Harbor or promote the dataset to executable
readiness.
