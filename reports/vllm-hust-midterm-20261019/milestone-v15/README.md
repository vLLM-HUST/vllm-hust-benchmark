# V1.5 课题四方法与复用 / Project 4 method and reuse

当前依据为《项目里程碑测试大纲》V1.5课题四，不是旧独立大纲V5.27或终期要求。 原文摘录、JSON源索引及提供方manifest逐字节保留。源DOCX未在本仓库提供；其摘要来自提供方，
不能把摘录校验说成独立验证了源DOCX。源文件三项SHA256均与交接值一致。

The current authority is Project Milestone Test Outline V1.5, Project 4. The exact extract and
source indices are preserved, together with the supplied manifest. The source DOCX is not included;
its digest is supplied provenance, not an independent verification of that DOCX here.

- [Exact text](project4-v1.5.txt), [source indices](project4-v1.5.json), [manifest](manifest.json).
- [Current M01 method/status](method-status.json), [offline reuse audit](reuse-audit.json).
- CPU-only rebuild: `python scripts/build_milestone_v15_status.py` from repository root.

## Current boundaries

B0 is always vLLM 0.18 / vLLM-Ascend 0.18. B1 has no base-version restriction; identify actual
versions and optimization configurations. Use the same base model/source weights, tasks, request
semantics, sampling/output constraints, quality/SLO, statistics and comparable physical resources.
Quantization, caches, speculation, graphs, scheduling and logical parallelism may differ. Do not
weaken B0, tune during execution, or splice different configurations' best metrics.

M01: C=F/(k×T×10¹²), eta=T_reference/T×100%, threshold 50%. Exclude padding, redundant work,
unaccepted drafts and cache-omitted work from F. Use matched-workload uninstrumented synchronized
wall clock; no estimated instrumentation subtraction. Derive the reference independently using
actual dependencies, overlap and necessary scheduling, not global max(resource times) or reverse
fitting. The new communication graph reference is available; model-level attribution remains open.
Current C/eta/pass remain undetermined, not zero and not the historical instrumented 30.25%.

Performance: throughput gain >=10%, request-weighted mean TTFT and TPOT improvement each >=5%, with
applicable quality/SLO. C16 fixed-B0 throughput is existing evidence, not a new test or a full
acceptance claim. Development-stack pairs retain their actual comparator identity.

Stability: cover the five specified task scopes, anomalies and actual >=16K continuous intervals,
with comparable 4K load and window P95/P99. Compatible 32K records can be reused. Six synthetic
retrieval probes do not prove full-task continuous stability.

Cost: allocate monthly all-in cost P across all occupied cards k and measured seconds T using
P×k×T/(24×D×3600); cost per million output tokens is 10⁶×expense/N. Input, cache-hit input and
output usage must refer to the same requests initiated and successfully completed inside the
interval. Cross-interval usage is separate. Compare time-weighted, per-card AI Core utilization over
the same interval. Fixed fees receive no utilization discount. The gate is >=10% cost reduction plus
clearly improved high-load utilization; full lifecycle cost is historical only.

## Decision-relevant gaps

| Requirement | Reusable evidence                                                          | Current decision / minimum remaining work                                                             | Responsibility                                                    |
| ----------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| M01         | Model counts, timing audit, measured 75-shape graph reference              | Not determined; useful-work attribution, matched synchronized timing and model dependency reference   | Evidence/recalculation thread                                     |
| Performance | Fixed-B0 C16 token-window audit; existing quality and development controls | Confirm C16 client/resource identity and applicable fixed-B0 latency/quality/SLO; no blanket repeats  | Evidence/recalculation thread                                     |
| Stability   | Existing task logs, long-input probes and synthetic windows                | Consolidate five-task anomalies and qualifying continuous windows; not six-probe acceptance           | Evidence/recalculation thread                                     |
| Cost        | Historical request resource coefficients and card samples                  | Fee basis, representative fixed-B0 pairing, same request population and weighted interval utilization | Evidence/recalculation thread; fee basis from environment support |

No new hardware runs, original-file search or changes to active tasks are authorized by this audit.
