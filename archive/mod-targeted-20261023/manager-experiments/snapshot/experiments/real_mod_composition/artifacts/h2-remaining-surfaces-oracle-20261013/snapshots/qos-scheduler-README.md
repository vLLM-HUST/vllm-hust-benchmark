# QoS Scheduler for vLLM-HUST

Owner-led research carrier for deadline-aware TTFT, TBT, and TTLT scheduling. It is a scheduler policy, not a deployment system or control plane.

**Status: request-level SLO validation, monotonic deadline tracking, and ordering keys are installable and tested; vLLM API/scheduler attachment remains blocked until `HOST_CONTRACT.md` is implemented.**

Technical ownership belongs to @Yushuo-star. Source extraction must preserve exact authorship, license, tests, constraints, and evidence before activation is considered.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Extension framework

Extension ID: `org.vllm-hust.qos-scheduler`

This repository follows the vLLM-HUST Extension Template. The current package
is deliberately `import_only`: it can be built, installed, discovered, and
inspected, but Extension Manager must refuse enablement until the maintainers
land a real host contract, implementation, compatibility evidence, and tests.

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.qos-scheduler
vllm-hust-ext extension check org.vllm-hust.qos-scheduler
pytest -q
```

The static Manifest 0.2 descriptor lives inside the Python distribution under
`src/`. Installation alone changes no vLLM behavior.
