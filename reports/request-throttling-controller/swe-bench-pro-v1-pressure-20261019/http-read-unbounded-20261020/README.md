# Approved Agent HTTP read-timeout isolation: start snapshot

**Native is running; there is no terminal performance or quality claim.**

After explicit user approval, both arms use registered `read-unbounded-v1`: only the Agent HTTP read
timeout changes from 180 seconds to no separate read timeout. Connect/write/pool stay at 180
seconds. Downstream transport remains 900 seconds; each Agent remains limited to 1200 seconds, 30
steps, 4096 output tokens and the 32768 context window. The retained 256-sequence / graph256 model,
KV16GiB, tasks, containers and official scheduler are unchanged. No model/proxy restart occurred.

The real workload is 8 official V1 tasks ×32 independent containers/Agents/programs, 256 at time0
without client slots, preserving program dependencies. Native started at epoch1792475849.1265664
with256 unique containers; Controller had not started at this snapshot.

The frozen client CPU mock gate verifies request extensions connect/write/pool180 and readnull with
one mock attempt, zero network/model requests, and JSON-serializable official model configuration.
Actual warmup separately proves graph replay increments4/4 on both ranks.256Native HTTP receipts
and256valid live trajectory files were read back, with zero observer errors. These are
configuration/start witnesses, not successful Agents.

The initial r1 preparation was rejected before submitting Agents because official save_traj cannot
serialize an HTTPX object inside model.config; zero Agent/model requests, retained. r2 uses the
official model.query keyword override API, retaining serializable baseline configuration and
separately recording effective request phases in each Agent receipt. Official Agent and ThunderAgent
files are unchanged. Source5c81661,27targeted tests and151full tests passed,4skipped. Both
cards/model remain retained.

The included report is an immutable start snapshot. Any later terminal result must retain errors,
unequal work and all budget boundaries; no repeated failed controls or gain claims from error exit
times.

## Terminal result: removing the read limit did not yield completed Agents

The original report.json remains the immutable start snapshot. terminal-report.json records the
completed first Native/Controller pair. Both arms used256unique independent containers,8official
tasks times32replicas, time0without client slots, and the approved read=null profile. Model/proxies
were retained without restart; all other budgets stayed fixed.

| Observation                                                |     Native | Controller |
| ---------------------------------------------------------- | ---------: | ---------: |
| Submitted                                                  |      0/256 |      0/256 |
| RetryError                                                 |        256 |        256 |
| Exception explicitly contains1200-second wall-budget alarm |        256 |        254 |
| Separate remote disconnection                              |          0 |          2 |
| Saved responses                                            |       2170 |       2110 |
| Saved output tokens                                        |     121793 |     153594 |
| Physical KV peak                                           | 99.876238% | 69.801980% |
| Engine preemptions                                         |         14 |          0 |

The alarm raises TimeoutError inside the synchronous response-header read; HTTPcore/HTTPX map it
into ReadTimeout and OpenAI/LiteLLM/Tenacity wrap it. Therefore the exception type alone does not
show the removed180-second read limit firing. The exact message is
`Frozen Agent wall budget reached`. Two Controller chains instead show a server disconnection and
are retained separately. No arbitrary budget increase or further pair was launched.

Error exits take1220.104799/1219.942621seconds including cleanup; neither is a successful batch
makespan. Work differs, zero Agents submit, all512grades remain unresolved, and no performance
benefit is established. Controller pause/mark/resume log-suffix counts908/918/899 are events, not
unique programs. Both arms begin/end with running/waiting zero. The lower physical pressure does not
prove why individual programs exceed the total budget. Non-streaming evidence still lacks individual
queue/pause/generation decomposition.

Archive52154730bytes/8183files passed independent whole/per-file hashes. NPU logs are separate.
Native start/intermediate/boundary and final live evidence were retained, but Controller
start/ten-minute live UI captures were missed during observation interruption; the per-second saved
CPU observations and terminal UI do not imply continuous live-screen coverage. Standard PNG/PDF
plots use the existing palette/layout and10-second rate, with hybrid KV token-equivalent rather than
exact resident tokens. Cards/model remain retained.
