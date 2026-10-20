# HCCS 64-bit Counter Evidence

The current Microflow B1 configuration completed the unchanged 200-request MMLU-Pro cohort: 282,265
input tokens, 51,200 output tokens, zero failures. The model controller reserved all four cards and
released them after the run. This hardware diagnostic does not replace the uninstrumented B1
selection.

## Measurement

- Read-only `dcmi_get_device_info`, HCCS main command 16, subcommand 5.
- Physical card 4 / logical 0 and physical card 5 / logical 1.
- Only the first two 16-port arrays are decoded. The installed driver returns 896 bytes; the newer
  header's full tail layout is not assumed.
- Sampling is approximately 50 Hz. The first-to-last request span is 62.89518077392131 seconds; the
  counter boundaries extend approximately one second on either side and are stable over their
  checked neighborhoods.
- Card 4 port 4 sends 10,381,085,319 flits, exactly received at card 5 port 1. The reverse direction
  carries 10,345,074,440 flits. Other ports have zero deltas. There are no negative intervals.
- Counts include incidental protocol traffic. Do not add TX and peer RX as independent traffic.
  Physical byte conversion and M01 attainment remain null.

The vendor documents HCCS port statistics in
[flits](https://www.hiascend.com/doc_center/source/en/mindcluster/730/clustersched/schedulingug/dlug_api_005.html).
No unsupported fixed byte-per-flit conversion is applied.

## Qualification

Each rank executed 256 all-reduce and 256 all-gather calls with 16 MiB BF16 input per call. Both
phases passed numeric checks. Settled peer TX/RX deltas match exactly, all counters are monotonic,
and final idle increments are zero. Counter refresh trails completion: immediate request-boundary
subtraction can miss traffic. The paced probe is not a peak-bandwidth calibration.

## Reproduction

Verify `SHA256SUMS`, concatenate archive parts in `archive.json` order, and verify the concatenated
SHA-256 and all member hashes in that manifest. The archive includes raw counters, request results,
model and device logs, resource-release receipts, qualifier and integration scripts.

To rerun the offline qualifier, place the archived `source/qualify_hccs64.py` beside the archived
run directories. The model auditor also imports `request_phase` from this report's
`recover_model_hardware.py`; set its import directory to the local checkout. Derived JSON outputs
must be written to new paths rather than overwriting archived evidence.

The archive also records a gold-label recovery for the 200 input case IDs: all IDs and frozen token
counts match the earlier 12,032-task MMLU evidence in
`reports/mmlu-pro-qwen35-b0-precheck-20261012/results/per-case-outcomes.csv.gz`. This is input
preparation, not a new quality measurement.
