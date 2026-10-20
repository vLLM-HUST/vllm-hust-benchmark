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
