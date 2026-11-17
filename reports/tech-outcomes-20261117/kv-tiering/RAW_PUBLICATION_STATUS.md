# Evidence access status

This draft publishes allowlisted numeric derivatives, configurations, frozen source identities and
an offline recomputation helper in Git. See [public/README.md](public/README.md). Public
recomputation covers all nine complete arms (9,885 requests), including cold rounds and
zero/negative results.

The three complete original archives remain byte-preserved on the custodian's local filesystem.
[controlled-location.json](controlled-location.json) identifies `JieYang2001`, the
workspace-relative locations, sizes, hashes and the procedure for a project reviewer to request
controlled access via [PR #697](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/697). Local
availability and hashes have been checked; delivery to an independent reviewer has not been
verified. There is currently no permanent remote download endpoint. A custodian index is not proof
of third-party access or acceptance.

Complete prompts, trajectory contents, token arrays and generated SSE/text are not included in the
public numeric export. The export can reproduce reported timing/counter arithmetic but cannot
independently verify the hidden response contents or task quality. The existing complete-SSE audit
remains available for controlled verification.

A Release is not the sole evidence route: issue #664 explicitly permits auditable manifests and
controlled-location indices for originals that require controlled storage. No Release/tag or binary
attachment has been published. This replaces the earlier overly restrictive requirement that this PR
must wait specifically for a public Release; it does not declare the draft accepted or change its
state.
