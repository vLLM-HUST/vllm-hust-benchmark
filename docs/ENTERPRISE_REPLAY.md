# Enterprise replay

All eight delivered JSONL assets are registered outside Git by logical path, record count, SHA-256, source format, provenance class, and workload-family mapping.

The evidence classes are intentionally different:

- `enterprise_observed_request`: real request bodies supplied by the enterprise. They provide credible API and request-shape evidence.
- `enterprise_provided_synthetic_or_hybrid`: controlled long-prefill, prefix, conversation-reuse, and semantic-similarity workloads supplied with the enterprise package. They provide useful stress and grouping semantics but are not presented as native production traces.

Neither class alone proves an engine cache hit or performance gain. Those claims require candidate/baseline measurements plus applied/effective engine telemetry.

## Replay modes

The loader preserves prompts and OpenAI request fields. The runner may apply an explicit benchmark normalization overlay for generation parameters such as `max_tokens`, `temperature`, and `seed`; the overlay is stored in the resolved spec and redacted record. It is not silently injected from a filename or global default.

For datasets carrying trace-shape fields, the authorized in-memory object preserves `group_id`, `session_id`, `conversation_id`, `created_at_ms`, `prompt_hash`, `prompt_tokens_est`, and an allowlisted subset of aggregate metadata. Persisted results hash group/session/conversation keys and never store prompts, messages, tools, or complete request bodies.

Sampling is request-based for independent data and group-aware for prefix or conversation data:

- `prefix_shared`: select complete `group_id` groups and replay in group/timestamp order;
- `reuse_conversation`: select complete `conversation_id` groups and replay in group/timestamp order;
- `semantic_similar`: select complete semantic groups, but do not treat semantic similarity as exact-prefix evidence;
- `long_prefill`: select requests deterministically and replay by source timestamp.

## Mandatory gates

1. Pass `--data-root` or set `VLLM_HUST_ENTERPRISE_DATA_ROOT`; no repository-adjacent path discovery is performed.
2. Provide a dataset-scoped technical authorization file. This file is a fail-closed runner gate, not a substitute for project-level data authorization, retention, and deletion approval.
3. Verify the entire checksum and non-empty JSONL record count before parsing requests.
4. Freeze case ID, sampling unit, seed, selected request/group IDs, replay order, and any generation normalization.
5. Keep source-model identity as provenance and apply the served model only at submission time.
6. Persist redacted shape metadata and runtime results only.

Authorization file example:

```json
{
  "schema_version": "enterprise-data-authorization/v1",
  "authorized_dataset_ids": ["<dataset-id-from-registry>"]
}
```

List cases:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.enterprise_replay --list-cases
```

Validate, sample, and emit redacted records:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.enterprise_replay \
  --case-id <case-id> \
  --data-root <authorized-data-root> \
  --authorization-file <authorization.json> \
  --served-model <model-repo-id> \
  --limit 100 --seed 0 --output redacted-records.json
```

The Python API returns request bodies in memory only to the authorized runner. Call `to_openai_payload(..., generation_overrides=...)` immediately before submission and persist `redacted_record(...)` plus runtime metrics.
