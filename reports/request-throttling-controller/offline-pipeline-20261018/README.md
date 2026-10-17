# Unified offline pipeline delivery

Root implementation/review: https://github.com/vLLM-HUST/request-throttling-controller/pull/3 , commit ebb6f5f. The configurable entrypoint reuses standard RIDE payload preparation, Ray Jobs, the canonical trajectory replayer, collection/export and paired summary tools. Six regression tests pass. It defaults to all-at-time0/no-client-cap input and keeps program dependencies. Unknown submission outcomes and stale cross-run collection are refused.

Real terminal reconciliation: finish mode queried all four actual Ray jobs, verified their private archive/report/seed bindings and public capsules, and reproduced the published three-seed summary exactly. No measurement was resubmitted. Source snapshots and sanitized terminal receipts are included; machine addresses and the private config are excluded. Fresh submission reuses the existing real-machine RIDE/Ray route and has configuration/generated-code tests; no extra model run or provisioning was performed solely for this packaging step. Optional platform provisioning hook therefore has no new real provisioning acceptance claim.

The measured output-rate ratio median is1.916861275x for the explicit APC-on/512MiB experimental profile. Full-batch makespan, task accuracy and original profile qualification remain unmeasured. Website publication is tracked separately from code/data review.
