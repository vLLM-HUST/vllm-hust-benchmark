# Real R4 source-drained attention observation

One fresh Qwen 7B TP2 request on NPU 3/6 returned HTTP 200 and passed the
frozen task check. Its complete unfiltered journal has 261 records, including
56 tensor artifacts, 56 cached native argument-completion records, and two
source-owned worker closures. The authenticated final ACK covers all 261.
Original owner exit was 0; watchdogs did not fire and owned processes were cleared.

The archive contains all receiver tensor bytes, raw requests/responses, status
polls, source snapshots, observed epochs and cleanup records. Split parts form
one complete XZ archive; the manifest binds every part and all 300 members.
Relocation and other late observations are explicitly separate from preregistration.

Run from the repository with Python standard library only:

    python -I -S experiments/formal_campaign/verify_attention_source_drained_observation.py --archive-dir experiments/formal_campaign/artifacts/attention-source-drained-observation-20261014-r4

The helper runs the captured consumer with site packages disabled and denies
access to original /root, model, device and process paths. It verifies recorded
evidence; it performs no fresh OS authentication or hardware execution.

This is finite nonformal evidence, not complete physical KV equivalence, numerical
attention correctness, full model quality, performance, or formal H3/H4. R3
remains a failed complete-prefix qualification. H1 remains falsified, H2 independent
n=4, formal H3/H4 remain 0, and submission status remains NO-GO.
