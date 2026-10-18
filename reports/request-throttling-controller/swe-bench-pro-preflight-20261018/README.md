# SWE-bench-Pro real-task preflight: blocked before repair

Conditional go for a real software-repair experiment, but no end-to-end task completed. This capsule
is a source/environment/interface smoke, not a repair score or scheduling benchmark. Actual SWE
tasks evaluated:0; resolved count and success rate are unknown, not0%.

Pinned official ScaleAI SWE-bench-Pro V2 commit66f92766bba642462d4bbe5479e83f91f9211862:642
tasks,6436 upstream file checksums PASS. Official Harbor0.24.0 Task loader accepted all642;
locked_mini_swe, patch_replay and host_mini_swe imports PASS. The task selected before outcomes is
the lexicographic first ID. Its public image digest/platform is recorded:linux/amd64 only. Existing
platform nodes were all arm64; local Docker was absent. A separate existing SSH route closed the
connection. No machine was reconfigured and no paid external sandbox was allocated.

Actual official `harbor run -p <first-task> -e docker -n1 -k1 -r0 -a oracle` and the same command
with `-a nop` both exited1: Docker is not installed or not on PATH. No sandbox, agent repair,
reference/empty grade, or leakage probe ran. These infrastructure failures are not failed
task-quality samples.

Two isolated UID-bound named Ray jobs used the retained idle Gold same-node TP2 Qwen3.5-35B-A3B BF16
service without restarting engines or touching Revision. Each made one Native and one Controller
request with the official mini-swe-agent bash tool schema. Default auto choice:both HTTP400, missing
auto-tool-choice/parser. New frozen named-bash profile:both HTTP400, parser still required in the
actual installed runtime. This attempted workaround preserves the sole required tool but differs
from the official default; its failure remains. All four API requests had no generation usage or
tool call. Controller program release was acknowledged in both jobs. Private request/response bytes,
matching forwarded requests, plan, release and hashes were independently checked; two original
private archives are bound below. No raw prompts/answers/endpoints are public.

Next:own an amd64 Docker CPU sandbox; verify exact-digest oracle/empty and official network/history
probes. Use a new serving profile with real tool-call support, freeze official
Agent/model/tools/time/Token budgets and lifecycle mapping, then complete one real patch and
pristine-sandbox regrade before measuring an uncapped time0 finite batch. Do not change the running
dataset engines to repair this preflight. No website score is added. Public verification checks
capsule consistency and hashes; it does not rerun private archives or prove task resolution.

Sources:
[official Pro V2](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2),
[vLLM tool-call modes](https://docs.vllm.ai/en/v0.23.0/features/tool_calling/). Documentation
suggests named function calling is enabled by default; the retained actual runtime response is the
governing evidence for this deployment.
