# vLLM-HUST host-contract maintainer acceptance — 2026-10-10

This record establishes human acceptance of the current vLLM-HUST typed plugin
contract line. It is not acceptance by the upstream `vllm-project/vllm`
project, and it is not native-NPU runtime qualification.

Current host main is `e020cfe3c4a675fc984748fc70ae6e201f1d0a4b`.
The following independently authored contract changes are ancestors of that
commit and remain present after the latest upstream synchronization:

| Contract | Author | Maintainer acceptance | Merge commit |
| --- | --- | --- | --- |
| Request KV materialization hooks, PR #20 | `healer-positive` | merged by `ShuhaoZhangTony` | `6baa026f602fd221dc7db64362730305048a8b87` |
| Request lifecycle events, PR #6 | `xiehanlong834-gif` | merged by `ShuhaoZhangTony` | `7620b23ab6d91230ff1c3f65f2dc4727fdec90c9` |
| KV offload observer seam, PR #46 | `xiehanlong834-gif` | merged by `ShuhaoZhangTony` | `115e5f6b0d2cb6f832293859e443cf9470201a21` |
| KV observer v2 correlation, PR #48 | `Remygred` | merged by `ShuhaoZhangTony` | `5a820400f65612f7564d22551b1c0433c70dbd18` |

GitHub records no separate submitted review on these PRs. The evidence is the
combination of a non-maintainer-authored change, explicit maintainer merge, and
survival on current main. Current main's pre-commit workflow passed in run
`38021724178`.

The current tree retains `vllm/plugins/request_processing.py`,
`vllm/v1/core/kv_materialization.py`,
`vllm/plugins/extension_capabilities.py`, and both offloading observer modules.
Any future upstream sync that removes or incompatibly changes these contracts
requires a new acceptance record and support-matrix update.

This closes only the human host-contract acceptance gate. It does not prove
serving correctness, process-owned effect, rollback, device release, or
performance on the current native NPU host.
