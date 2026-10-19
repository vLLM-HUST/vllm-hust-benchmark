# SWE-bench Pro V1 ARM: real-Agent Native/Controller comparison

The reusable repository entry actually ran the same8 fixed V1 NodeBB tasks in six batches, NC/CN/NC,
on one unchanged unified HUST graph-mode model epoch. All tasks arrive at time0; there is no client
slot cap, and every program retains its model/tool dependencies.

| Arm        | Median Agent-batch seconds | Saved model responses, three batches | Output tokens | Terminal statuses                                                       |
| ---------- | -------------------------: | -----------------------------------: | ------------: | ----------------------------------------------------------------------- |
| native     |                 365.013276 |                                  679 |        155294 | {'LimitsExceeded': 18, 'ContextWindowExceededError': 5, 'Submitted': 1} |
| controller |                 351.572028 |                                  682 |        159424 | {'Submitted': 4, 'ContextWindowExceededError': 5, 'LimitsExceeded': 15} |

Each arm has official passed-patch counts1/8,0/8,1/8 across the three repeats. These are repeated
executions of the same8 tasks, not24 distinct tasks. Passing saved partial patches and clean Agent
Submitted exits are separate outcomes; their actual termination states are retained in report.json.

These are bounded workflow measurements, including tool waits. They are not pure decode TPS, three
independent model-start repetitions, a full quality leaderboard, or full deployment cost. Saved
response usage does not assign zero tokens to failed calls without usage. Card-seconds cover the
Agent window only; monetary price is unknown. Work/errors and actual scheduling events must be
considered before attributing a time difference to the MOD. This capsule makes no benefit claim.

The original batch grading wrapper missed required official Dockerfiles and cwd. Its false values
are retained as infrastructure failures, not repair results. A separate frozen grading entry ran
only the saved patches after all Agent batches; the model was not rerun. Independent verification
replays the unchanged official parser, checks original patch/source/image bindings, new Docker
container events, all48 task usages/timings/releases, both workers' actual graph replay and epoch,
and private archive hashes. Missing grading outputs remain unresolved.

The retained first-task reference patch also has four strict test-name mismatches. No test names,
scoring rule, task, model, history, dependency or generation budget was changed to rescue scores.
Pro V1 is sufficient for the engine-performance objective; no V2 or x86 campaign is required.

The private archive contains raw prompts, responses, patches, tests, addresses and original logs;
none are published here. `verify.py` checks public capsule integrity and arithmetic only; it does
not independently access or reverify the private archive. Frozen execution snapshots under
`sources/` retain exact bytes and must not be formatted.

Reuse:
[repository entry and deployment instructions](https://github.com/vLLM-HUST/request-throttling-controller/blob/codex/offline-batch-experiment-contract/docs/experiments/SWE_BENCH_PRO_REUSE.md).
Sources:
[official V1](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862),
[official Agent](https://github.com/scaleapi/mini-swe-agent/tree/d74716a3c8104a113f77cc9ab94cf407ecdcf1e9),
[ARM recipes](https://github.com/bdqnghi/swe_benchmark_arm/tree/b2834f1b0cf94294f409e93c7dccc398b0a2699b).
