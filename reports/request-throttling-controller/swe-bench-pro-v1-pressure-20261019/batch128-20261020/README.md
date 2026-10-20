# Batch128: real Agent first-pair terminal result

**Neither arm submitted an Agent. No batch performance benefit established.**

The user requested a smaller batch after256-program budget failures. This run uses8official V1
NodeBB tasks times16independent ARM containers/Agents/programs,128at time0without client slots. It
preserves per-program dependencies and changes only batch size and replica count. The retained model
remains max_num_seqs256/graph256/KV16GiB/context32768;
Agent1200seconds/30steps/4096output/readnull/connect-write-pool180/downstream900 and the fixed
official core/capacity adapter remain unchanged. No model/proxy restart or additional64-program scan
occurred.

| Observation                              |     Native | Controller |
| ---------------------------------------- | ---------: | ---------: |
| Submitted                                |      0/128 |      0/128 |
| 1200-second wall-budget exception chains |        128 |        118 |
| 30-step limit                            |          0 |          5 |
| Context-window exceeded                  |          0 |          5 |
| Saved model responses                    |       1540 |       1865 |
| Saved output tokens                      |      87405 |     152924 |
| Physical KV peak                         | 67.574257% | 66.584158% |
| Engine preemptions                       |          0 |          0 |

Error exits take1208.844020/1208.966642seconds; these are not successful batch makespans.
All256grades remain deferred/unresolved, not zero scores. Five Controller step-limit records have
exactly30model calls.246exception chains explicitly contain `Frozen Agent wall budget reached`,
whose TimeoutError is mapped into ReadTimeout by the HTTP stack. Do not interpret it as the
removed180-second read timeout firing. Both arms begin/end with running/waiting zero. The smaller
batch avoids KV saturation/preemptions in these samples but still does not complete within the fixed
constraints; this does not identify each request's queue/pause/generation time.

Controller pause/mark/resume counts1015/1000/998 are events from an exact retained-log suffix, not
unique program counts. Raw archive40209726bytes/4338files passed independent whole/per-file hashes.
NPU logs are separately retained. Start, a Native six-minute checkpoint, and later terminal UI are
saved; later Native/Controller intermediate and transition live screenshots were missed and are not
claimed. Per-second CPU observations remain. Standard figures reuse the existing plot_run
palette/layout and10-second rate; hybrid KV is token-equivalent rather than exact resident tokens.
Both cards/model are retained.
