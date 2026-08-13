# HUST Science Serving Benchmark v1 proposal

NatureBench is used as an authoritative design reference, not copied or treated as a cheap serving benchmark. Its paper-sourced tasks, executable environments, hidden evaluators, validity checks, and six-domain coverage motivate a smaller benchmark built specifically for inference-engine acceptance.

## Scope

`hust-science-serving-v1` contains 24 independently licensed task packages: four tasks in each of cellular omics, protein biology, biomedical modeling, physical modeling, molecular design, and relational reasoning.

Each package must include:

- a scientific problem statement and data-description file;
- a fixed container image digest and data checksums;
- a bounded compute and wall-clock budget;
- a deterministic submission schema and hidden evaluator;
- one primary task-quality metric;
- validity checks that reject empty, hard-coded, leaked, or malformed submissions;
- a captured sequence of LLM calls with tool time excluded from engine-service metrics.

Tasks are newly constructed from open, redistribution-compatible scientific assets. NatureBench task bodies, hidden evaluators, and restricted data are not copied without explicit license permission.

## Two-channel evaluation

The quality channel runs the complete agent loop once per admitted stack and scores all 24 tasks. It establishes that the engine still supports successful scientific work.

The performance channel replays the frozen LLM segments extracted from valid quality runs. It fixes prompt bytes, tool schemas, generation parameters, and request order, then compares stock, feature-off, feature-on, and minus-one configurations. External tools, container execution, scientific training, and network time are excluded from inference-engine performance attribution.

## Model and gates

The primary model is the pinned Qwen3-Coder-30B-A3B-Instruct artifact used by the code scenario. Reusing it limits deployment cost and matches the executable coding/verification task form.

Release gates are:

1. six-domain smoke: one task per domain loads, executes, and produces a valid evaluator result;
2. task-package gate: all 24 packages pass license, checksum, container, determinism, and hidden-evaluator review;
3. quality gate: task-success count is non-inferior to feature-off by no more than one task;
4. serving gate: frozen LLM-segment output-token throughput improves by at least 5%, with a paired-bootstrap 95% confidence interval excluding zero;
5. protocol gate: zero malformed tool calls, parser failures, silent fallbacks, or unclassified engine errors.

The six-task smoke is compatibility evidence only. It is not the final AI4Science quality result.
