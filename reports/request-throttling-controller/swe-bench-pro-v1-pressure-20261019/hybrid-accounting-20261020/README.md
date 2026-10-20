# Hybrid cache accounting: real Agent first-pair terminal result

**Controller reduces physical KV pressure and preemptions; every Agent still fails. No performance
benefit established.**

The workload is8official V1 tasks ×32independent containers/Agents/programs,256at time0without
client slots. Program dependencies remain. Both arms reuse the retained approved
max_num_seqs256/graph256model, KV16GiB, official pinned scheduling core,5-second scheduler/metrics,
HTTP180seconds,1200-second Agent budget,30steps and4096output tokens. No model restart occurred in
this phase.

The existing audited capacity adapter changes usable capacity1656832→1654784(excluding the null
block) and per-program upper reservation100→14335. Actual frozen SDK reconstruction matches809blocks
and the1204968startup estimate, with3Mamba groups and1attention group. The reservation is6fixed
Mamba upper-bound blocks plus2047attention rounding tokens. It is not a universal exact
resident-token model or an arbitrary smaller KV allocation.

| Observation                  |  Native |    Controller |
| ---------------------------- | ------: | ------------: |
| Peak physical KV             |    100% |    65.841584% |
| Engine preemptions           |      13 |             0 |
| Pause / mark / resume events | 0 /0 /0 | 688 /697 /662 |
| Submitted Agents             |   0/256 |         0/256 |
| RetryError                   |     253 |           229 |
| LimitsExceeded               |       2 |            20 |
| ContextWindowExceededError   |       1 |             7 |
| Saved responses              |    1959 |          1714 |
| Saved completion tokens      |  116859 |        143187 |

Native/Controller finish with errors after1005.274888/1115.241714seconds. The work differs and no
Agent submits: these are not successful batch makespans. All512grades remain deferred/unresolved,
not zero scores. Running/waiting are0before/after both arms. Both ranks actually replayed graphs;
warmup deltas4/4.

All253/229ReadTimeout exception files respectively contain HTTP read timeouts. Capacity adaptation
changes physical pressure but does not eliminate timeout failures. The retained non-streaming
evidence does not split individual queue, pause and generation delays. Context-window failures are
separate: at least28673input tokens plus4096requested output exceeds32768. Event counts are not
unique program counts.

Raw archive38851627bytes/7649files passed independent per-file hashes. NPU logs are separate. Start,
intermediate and terminal platform Grafana/Ray/log receipts were retained; public artifacts contain
summaries, hashes, source and standard PNG/PDF figures only. Standard engine plots reuse the
existing palette/layout and10-second rate window; hybrid KV is token-equivalent, not exact resident
tokens. No repeat CN/NC was launched and both cards/model remain retained.
