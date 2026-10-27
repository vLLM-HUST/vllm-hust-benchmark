# Installation assets and legacy document follow-up

Scope: benchmark #326, 2026-10-24. This is custody and consumer review, not new performance
admission. No published software asset is deleted or replaced.

## Installation consumers

The [machine-readable review](installation-consumers-20261024.json) records paginated Release/asset
enumeration, downloaded sizes and verified SHA256 for nine assets in three public MOD repositories.
Archive inspection was read-only; no package, installer or experiment was executed. Filename
candidates are not automatically classified as measured data.

| Repository | Assets / consumer evidence | Disposition | | --- | --- | --- | | Extension Manager |
v0.3.0 wheel and sdist; wheel console entry point `vllm-hust-ext`, build/install consumer in
[pyproject](https://github.com/vLLM-HUST/extension-manager/blob/be30e724ea4b8fdae7449ea716bb7aee2666f8bd/pyproject.toml)
| Keep install artifacts; independently bind historical sdist evidence below | | Production Stack |
Six chart archives, 0.1.13–0.1.18, still named with matching digest in
[gh-pages index](https://github.com/vLLM-HUST/production-stack-hust/blob/5fc1c37f9e3b28958f88f42999dbc65ce2a99240/index.yaml);
[release workflow](https://github.com/vLLM-HUST/production-stack-hust/blob/84d999532cfedd6caa9d0b95404bfecc9dc92550/.github/workflows/helm-release.yml)
publishes packages with index | Retain all versions; indexed installation packages are not
experimental duplicates | | Metal | CPython 3.12 / macOS arm64 development wheel;
[installer](https://github.com/vLLM-HUST/vllm-metal-hust/blob/0faca52fc462b3cee166caa159ae4836212980a3/install.sh)
resolves Release wheel assets, but defaults to upstream repository | Retain; upstream default is not
proof of zero direct/fork/old-version consumers |

Production Stack README also points its Helm install at the upstream chart repository, not
necessarily this fork. The fork index provides an additional installation path; it does not
demonstrate how many live clusters use it. No filename-classified evidence members were found in the
six charts or Metal wheel; this is not a general binary security or runtime qualification claim.

## Manager historical sdist contents

The v0.3.0 sdist contains 25 selected evidence/output/context members: 10 match already archived
benchmark bytes; 15 differ from or are absent in the currently examined Manager snapshots and are
preserved under [manager-release-v030](manager-release-v030/snapshot/). The release tag has no
standalone LICENSE file; its
[pyproject license declaration](https://github.com/vLLM-HUST/extension-manager/blob/v0.3.0/pyproject.toml)
is `Apache-2.0`. No missing license file is invented or claimed to be archived. `evidence_bindings`
in the review supplies every member path, size, SHA and either an existing fixed URL or the new
repository path. These include historical qualification/blocker documents, deployment
preflight/observation records, derived metrics and paper example/schema context. The paper example
and schema are not measured results. Existing Git-source versions are not overwritten. This proves
byte custody of the selected members, not that every mixed prose statement in the 267-member sdist
has been migrated.

## vSpec document migration

Three complete historical documents and LICENSE are frozen in
[vspec-agentx-school/manifest.json](vspec-agentx-school/manifest.json): AgentX, SPEED-Bench coding
acceptance length, and School 0.28 local argmax. Preserve synthetic sampling versus natural
correctness, negative/near-zero outcomes and all original workload/version identities. Current B0
remains vLLM/Ascend 0.18; historical Native arms do not redefine it.

Existing vSpec capsules provide six AgentX run/target/AIPerf records and four SPEED-Bench
acceptance-summary/dataset-hash records. Those ten files are usable direct links, not proof that
every log/output claimed by the prose survives. In the pinned capsule scope, SPEED-Bench
`result.json`, `bench.log`, `server.log`, `serve_cmd.txt`, and `bench_cmd.txt` are absent for both
modes; AgentX `harness.log` is absent for both arms. Keep these twelve precise missing-member
boundaries rather than inventing original token counts or initiating reruns. School 0.28 already has
a separate fixed benchmark report/raw-run collection; its `summary.json` is the consumer, not a new
result produced here.

## Remaining work and review discipline

Other MOD mixed documents, legacy consumers and older software assets remain classified work. No
external-consumer absence claim is made. Existing hardware, review-runner and unrecovered-original
blockers remain; 33 probes are not pursued and GDN is not retested. Policy detection and baseline
are unchanged.

Skill review: custody/consumer follow-up | user's #326 migration and CI-green merge request | pinned
public source, downloaded packages and manifests above | E04/E31/E34/E37/E42/E47 | immutable
originals, no asset deletion, no hardware run, update both data navigation and issue body/receipt.
