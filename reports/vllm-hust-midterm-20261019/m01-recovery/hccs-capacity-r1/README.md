# HCCS Reference And Driver Byte Convention

## Test

TP2 on physical cards 4 and 5, BF16, 16 MiB per-rank input. Both all-reduce and all-gather execute
8,192 calls per repetition, with three repetitions each. All-reduce includes input fill, both
include dispatch and synchronization. There is no deliberate inter-call pacing. All numeric checks
pass. All four allocated cards are reserved, checked idle before launch, and released.

Settled cumulative TX/RX counters match exactly across peer ports in all six phases. Counter
boundaries are stable, and no negative intervals occur. Median aggregate TX rates are
1,875,338,838.6176152 flit/s for all-reduce and 1,882,610,908.7985241 flit/s for all-gather. This is
a 16 MiB synchronous reference, not a calibration matched to every model payload and graph.

## Unit Evidence

The installed `libdcmi.so` SHA-256 is
`2e89f3a4d28e0002bb81f3617cede498784fd9cd28b2d0ab0121250e05fea10c`. The archive contains the
read-only disassembly of `dcmi_get_hccs_link_bandwidth_info` and its two floating-point constants.

- At `0x54834`, the multiplier is 20.
- At `0x80f78`, the little-endian double is 1000.
- At `0x80f80`, the little-endian double is `2^-30`.
- The getter computes `count * 20 / (profiling_time_ms / 1000) / 2^30`.

The
[vendor API documentation](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_149.html)
labels bandwidth GB/s; the installed implementation above uses GiB/s. This observation is
version-specific, not a claim about all driver versions.

Read-only calls collect the official getter alongside cumulative counters. The sampled getter rate,
converted using `2^30`, divided by the cumulative `flit * 20 / execution_seconds` rate ranges from
0.9950009679577833 to 1.006420280583857 over both directions, cards and all six repetitions. The
getter covers approximately half of the interior phase: this is a rate cross-check, not complete
traffic integration. The count-to-byte convention is supported by the pinned implementation and this
experimental agreement.

## Model Projection

`model-link-accounting.json` derives driver-accounted link bytes from the unchanged prior
`hccs64-r1/summary.json`. It verifies an identical driver hash. Two-direction TX totals are
414,523,195,180 bytes, including protocol traffic. Peer RX is not counted a second time. No
independent physical-line capture, tensor-payload equivalence or M01 attainment is claimed.

The 16 MiB median aggregate rates convert to 37.5067767723523 and 37.65221817597048 decimal GB/s,
respectively. These rates cannot be substituted for the model's complete mixed-shape communication
reference.

## Reproduction

Check `SHA256SUMS`, then concatenate the archive parts in `archive.json` order and verify the
concatenated and per-member hashes. Raw counter/getter samples, both rank results, NPU state, exit
receipts, source and decoder evidence are included.

Place archived `source/audit_hccs_capacity.py` alongside the extracted run directories to reproduce
the summary in a fresh directory. The model projection also needs the prior `hccs64-r1`
model/control evidence. Never overwrite archived raw or derived JSON; use a new reproduction
directory.
