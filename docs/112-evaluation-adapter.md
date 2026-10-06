# 112 Evaluation worker adapter

`scripts/run-112-evaluation-job.py` is the benchmark-side worker command for the dev-hub 112
Evaluation API. It accepts only the worker's `EVALUATION_REQUEST_FILE`, `EVALUATION_ASSIGNED_NPUS`,
and `EVALUATION_OUTPUT_DIR` environment contract plus an administrator-owned `--schedule` file.
Request `metadata` and `source_url` never control executable arguments or environment.

The versioned schedule must explicitly approve each target and pin its registry status, source-spec
SHA256, model ID/path/revision, hardware, campaign identity, and load profile. The common section
pins the benchmark/core/plugin repositories, runtime Python, image digest, CANN/torch-npu versions,
and topology. Do not store tokens in this file. The illustrative shape is in
`docs/examples/112-evaluation-schedule.example.json`; replace every placeholder with audited local
values and protect the deployed file from requesters. The execution plan records both
`schedule_version` and the schedule file SHA256.

The adapter verifies the generated registry checksum, exact registry version, target/spec identity
and SHA256, assigned card count, repeat range, and clean Core/Plugin commits. A 40-character model
revision must match a local model reference or model Git HEAD. A 64-character local-model revision
must equal the content fingerprint of the actual top-level `.safetensors`, `.bin`, or `.pt` weights;
a config-only digest is rejected. It constructs a fixed `run-campaign-repetitions.sh` command, sets
formal independent-service environment values, and starts the runner in a new process group.
TERM/INT are forwarded only to that owned group. The worker retains the adapter log and bundle
SHA256; the adapter retains the campaign summary and raw attempt directories even when the runner
fails.

**The execution adapter is not an admission or publication gate.** A zero runner exit is marked
`UNVERIFIED`. After the dev-hub worker writes `worker.json`, seals the directory with
`BUNDLE_SHA256`, moves it to its final path, and records the terminal API job row, run the
independent verifier:

```bash
python scripts/verify-112-evaluation-bundle.py \
  --bundle /immutable/evaluation/artifacts/eval-... \
  --job-record /outside-the-bundle/eval-....json \
  --schedule /administrator/112-schedule.json \
  --attestation /outside-the-bundle/attestations/eval-....json
```

The verifier recomputes the worker-compatible tree digest (excluding only `BUNDLE_SHA256`), rejects
symlinks, binds the canonical request to the terminal job row, and verifies worker identity, NPU
allocation, exit code, schedule snapshot, execution plan, complete three-or-more repetition summary,
every artifact checksum, target/model/source/runtime/campaign provenance, and online server or
offline graph evidence. The `VERIFIED` attestation is always outside the sealed bundle and is
indexed by job ID plus bundle SHA256. It refuses to overwrite an existing attestation.

Only an external `evaluation-112-admission-attestation/v1` with `status=VERIFIED` may proceed to
publication. No target without explicit schedule approval can run. The 1/2/4-chip
communication-sensitive targets remain draft until their executable contract and independent service
evidence are approved.
