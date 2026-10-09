# PyramidKV Terminal-Bench 2.1 source and image audit

This report advances PyramidKV issue #8 and benchmark issue #254. It verifies all 89 task sources
and their image metadata at the frozen Terminal-Bench 2.1 revision. No image layers were pulled, no
sandbox or agent ran, and no task resolution score or compression benefit is reported.

The source is
[harbor-framework/terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1/tree/7131e4375048a0e408a8fb404b5f499d726b695b).
The upstream tree contains 1,101 ordinary files; every extracted file was checked against its Git
blob identity and size. All 89 tasks have their instruction, task configuration, Dockerfile,
verifier script and reference solution in that tree. File identities do not prove that scripts or
reference solutions execute successfully.

## Verified findings

- All 89 task configurations map to digest-verified **linux/amd64** manifests and image configs.
  Four tags resolve through image indexes. Tags are task-specific: 79 use `20251031`, six use
  `20260403`, and four use `20260430`. Never replace these with a common tag or the current latest.
- The 319 unique compressed layers total **16,699,830,105 bytes** (about 15.6 GiB). The largest
  individual image has **1,432,498,021 compressed bytes**. These are registry metadata totals, not
  measured disk use, uncompressed storage, or a requirement to pre-pull every image at once.
- Task configurations request 1–4 CPUs, 2–8 GiB memory, 10 GiB storage and zero GPUs. All allow
  internet access. Agent timeouts range from 600 to 12,000 seconds; verifier timeouts range from 360
  to 12,000 seconds. The exact values for each task are in the inventory.
- The repository root contains an Apache-2.0 license. That does not by itself clear every task,
  third-party dependency, downloaded asset or image for every use. Task-specific rights review is
  still open under the canonical program contract.

The current model host is ARM64 and has no usable Docker/Podman socket or supported Harbor sandbox
backend. Running these AMD64 task environments needs a suitable local or remote execution backend,
plus a route to the model API. The model currently listens only on localhost. No endpoint was
exposed, no paid remote backend was provisioned, and no architecture-emulation claim is made.

## Reproduce

Use Python 3.11 or later. These scripts use the standard library and never execute task code.

```bash
python scripts/fetch_pyramidkv_terminal.py --assets /path/to/terminal-assets
python scripts/audit_pyramidkv_terminal.py \
  --assets /path/to/terminal-assets --output /path/to/new-audit
```

The fetcher uses the report's frozen Git tree and original image receipts. It retrieves the source
archive and only image manifests/configuration blobs, by digest rather than mutable tag. Anonymous
Docker Hub rate limits apply. Existing files are checksum-checked and reused; a mismatch stops the
run. A partial retrieval can be resumed. `--source-only` and `--task TASK_ID` permit partial
retrieval, but the full auditor requires every frozen task. Original image observation timestamps in
copied receipts remain the original observation times; re-fetching a digest does not re-observe a
tag.

The offline auditor checks the complete source tree, task-to-receipt mapping, image index and config
digest chain, declared platforms, resource limits and compressed layer sizes. It emits deterministic
`inventory.json` and `summary.json`. Two independently generated audits are byte-identical; their
hashes are in `reproducibility.json`. The inventory is split into gzip parts of at most 25 tasks to
keep individual artifacts small. Concatenate the decoded arrays in filename order to reconstruct it.
The raw task contents, reference solutions and image environment values are not republished.

Local cached retrieval for all 89 images and a fresh public retrieval check are recorded separately.
Full repository validation passed: **1,822 tests passed, four skipped**; staged pre-commit passed.
The initial test collection failure (missing `PYTHONPATH=src`) is retained alongside the successful
rerun. The 35 new synthetic tests cover corruption, changed mappings, ambiguous platforms, unsafe
archive paths, credential removal on cross-host redirects and reuse of validated downloads. None of
these tests counts as task execution or semantic verifier calibration.

## Remaining execution work and ownership

PyramidKV issue #8 assigns MOD coordination to @Irisuko. Benchmark issue #254 coordinates canonical
rights and execution-contract review; this report does not imply those reviews have been approved.
An execution operator must provide a supported AMD64 fresh-sandbox backend and model connectivity.

Before collecting task scores, pin Harbor and agent/scaffold revisions, tool and network policy,
context handling, token/step/time budgets, attempts, failure accounting and verifier invocation.
Preserve each task's frozen environment and timeout requirements. Validate reference-solution and
failure behavior in fresh sandboxes. Then run matched Qwen3.5 B0/B1 trials with compression
telemetry, raw agent traces, verifier outputs and failures. A source audit or successful image
download cannot substitute for those runs. The applicability/optimization status is not promoted by
this report.
