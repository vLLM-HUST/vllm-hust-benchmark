# SWE-bench Pro V1 ARM: pressure preparation and actual 256-Agent start

This is an unfinished real-Agent experiment, not a completed performance result. The same eight
official V1 NodeBB tasks each have32 independent Agent/container/program instances, for256
executions arriving at time0 without a client slot cap. These are not256 distinct tasks and not
replayed trajectories. Frozen official Agent/scoring/image and unified HUST runtime are reused.
Model,16GiB KV,maxseq16,30steps,4096output and1200second Agent budget are unchanged.

The prior eight-task six-batch archive was read-only audited:2104 valid observations, no observer
errors, KV peak13.490099%, no sampled scheduler overflow or Controller program waiting. Engine
instantaneous request queueing is a different metric. Raw809blocks×2048=1656832 differs from the
hybrid engine-reported1204968token-equivalent capacity by37.500083%; this boundary is retained
without changing capacity to force a trigger. Linear projection is not pressure proof.

A model-free CPU preflight actually prepared128 independent ARM containers and imported the fixed
Agent in57.944035seconds, with21.008476GiB dev peak and zero remaining owned containers. The
private94915byte archive has266 independently checked files; its hash is in report.json. This is
neither a128-model experiment nor peak real-tool resource validation.

The new model was started through the existing RIDE entry on the owned two-card worker. Both logical
devices executed a real tensor operation. Both actual model ranks recorded four new graph replays
before Agent startup. The first256-instance Native batch actually started; 256 different container
IDs were read back. Initial metrics show15 running and240 waiting requests, KV7.797%; queueing alone
is not KV saturation. Native then Controller is the first pair; additional pairs require an actual
pressure/mechanism/work audit. Both cards remain allocated under the user's explicit instruction.

Private raw prompts, responses, patches, network addresses and credentials are excluded. Actual
Grafana screenshots, Ray UI and raw logs are retained in the original Feishu document; the attached
Ray screenshot shows actual batch startup only. Standard final plots will reuse plot_run and
plot_swe_bench_pro after terminal data is available. Public verify.py checks only public hashes,
arithmetic and claim boundaries; it does not reverify the private archive. All failed attempts and
the previous no-action result remain retained in earlier capsules.
