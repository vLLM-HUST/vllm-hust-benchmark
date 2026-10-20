# Approved 256-sequence profile: real Agent first-pair terminal result

**Physical KV pressure and Controller actions observed; all Agents failed; performance comparison
invalid.**

After explicit approval, both arms used the new central pressure-256-v1 profile: engine max_num_seqs
and decode graph capture limits increased to256. Model, HUST runtime, KV16GiB, context32768, tasks
and Agent budgets stayed fixed. Actual two-rank graph replay increments were4/4. The workload was256
independent real Agents/containers/programs (8 official V1 tasks ×32), arriving at time0 without
client slots, preserving program dependencies. It was not256 unique tasks or replay.

| Observation                      |                           Native |     Controller |
| -------------------------------- | -------------------------------: | -------------: |
| Peak physical KV occupancy       |                       99.876238% |           100% |
| Engine preemptions               |                               15 |             11 |
| Paused / marked / resumed events |                          0 /0 /0 |  183 /207 /181 |
| Saved model responses            |                             2161 |           2196 |
| Saved completion tokens          |                           159277 |         121473 |
| Submitted Agents                 |                            0/256 |          0/256 |
| Agent terminal status            | 253 RetryError +3 LimitsExceeded | 256 RetryError |

Ray SUCCEEDED means the orchestration ended. Both batches ran about1200seconds until errors/budget
exhaustion; unequal saved work and zero submitted Agents preclude a valid performance gain. All512
grades are deferred/unresolved, not zero scores. Engine running/waiting were0 before and after both
arms, resolving the earlier pair's cross-arm residue boundary. Disconnect logs do not independently
prove every request's abort.

The original exception chain shows a timeout receiving HTTP response headers, wrapped by OpenAI,
LiteLLM and Tenacity. Agent HTTP timeout180s, downstream transport900s and total Agent budget1200s
are distinct. Exception files can contain both read-timeout and wall-budget chains; those counts
overlap.

At the first Controller sample with100% physical KV usage, ThunderAgent still reported1155488
remaining token capacity with no paused programs. All11 samples above99% had positive remaining
capacity. This proves an accounting disagreement at those instants; it does not uniquely explain
each timeout. The raw block-derived budget1656832 differs from the engine's group-aware startup
estimate1204968; the latter is not an exact universal per-length token budget either. No capacity
override was applied. A future correction needs verified group/fixed-state costs and separate
evidence, not an arbitrary smaller threshold to force actions.

CPU raw archive44321211bytes /7672files passed independent per-file hashes; SHA256 is in
`archive-verification.json`. Private model/proxy logs are separate. No private task text, answers,
patches, payload tokens, endpoints or credentials are published. Start/terminal UI evidence and
second-level observations were retained, with gaps in intermediate ten-minute screenshots explicitly
preserved. The included engine PNG/PDF files use the existing plotting style and10-second rate
window; hybrid KV is a token-equivalent estimate. The public verifier checks public
integrity/consistency, not private evidence.

The original r1 preinitialization TypeError and all previous failed runs remain preserved. Both
cards and the new model are retained. No repeated CN/NC pair was launched after this failure.
