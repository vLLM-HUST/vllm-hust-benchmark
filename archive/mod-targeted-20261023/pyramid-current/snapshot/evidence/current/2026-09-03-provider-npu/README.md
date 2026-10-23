# Current provider NPU selection evidence

This directory records a current-working-tree algorithm check on one physical
Ascend 910B2. It is deliberately narrower than a paired-host serving test: the
test compares the Qwen2.5-14B 40-query-head/8-KV-head grouped-GQA selection and
materialized K/V tensors against a frozen repeated-GQA oracle on the NPU.

It does **not** establish current vLLM-HUST plus vLLM-Ascend-HUST end-to-end
correctness, quality, capacity, or serving performance.

## Command

```bash
ASCEND_RT_VISIBLE_DEVICES=6 PYRAMIDKV_RUN_NPU_TESTS=1 \
  .venv/bin/python -m pytest -q tests/npu/test_selection.py
```

Result: `1 passed in 18.13s`.

The structured environment, source identities, command, and result are in
[`result.json`](result.json). The repository base commit is recorded because
this evidence was generated before the active-host changes were committed;
the three tested file hashes bind the result to the exact implementation and
oracle content.
