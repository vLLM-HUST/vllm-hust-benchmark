# MuxPD TP4 follow-up: ordinary workers, source deployment and bounded failures, v3

This append-only evidence candidate records new terminal diagnostics. It preserves v1/v2 numerical
evidence, the separately frozen quality protocol, and every failed attempt. The new ordinary-worker
replay matches the earlier observed outputs on the fixed input family. The source98 deployment
checks establish a bounded source-installation path. Neither observation establishes general model
quality, throughput, goodput or a grouped-prefill performance benefit.

All device facts below come from archived run receipts: Qwen3.5-35B-A3B BF16, TP4/EP4, four visible
devices 0..3 with physical 910B2 IDs 1/2/3/6. The shared native sources remain Core
`d0f22d2bda562156e4dbf433ce645e1769b4f804` and Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`.
The existing Ascend build-script modification and installed-binary provenance limitations remain
explicit; a source archive does not recreate or certify the installed binary.

## Fixed token continuity without the numerical observer

The f7 ordinary raw/group processes use exact production workers without the numerical extension or
sampler/runner observer. They preserve the frozen 12-phase composite, including the seed-D followed
by P384 discriminator, fixed prompts, eight output tokens, same cards, HCCL_DETERMINISTIC=true and
the same explicitly disabled npugraphEx/static-kernel backend. The group flag is the only raw/group
configuration difference. Each comparison below has 30 eight-token sequences, 240 token IDs, and 30
matching within-process warm pairs on each side.

| Comparison                                 | Exact output sequences | Explicit admission comparisons |
| ------------------------------------------ | ---------------------: | -----------------------------: |
| Observed raw 6c versus ordinary raw f7     |                  30/30 |            30/30 request pairs |
| Observed group 6c versus ordinary group f7 |                  30/30 |            30/30 request pairs |
| Ordinary raw f7 versus group f7            |                  30/30 |            30/30 request pairs |
| Ordinary raw f7 versus source98 raw        |                  30/30 |            30/30 request pairs |
| Ordinary group f7 versus source98 group    |                  30/30 |            30/30 request pairs |
| Source98 ordinary raw versus group         |                  30/30 |            30/30 request pairs |

The six retained comparison files were independently reproduced byte-for-byte. Authoritative
internal/external request mappings, returned prompts, completed outputs and all 12 phase admissions
were rechecked. These plain runs do not capture full logits, hidden states, actual peer batch shapes
or KV contents. They therefore extend the observed numerical study by an API-output continuity
check; they do not turn its limited captures into a universal equivalence proof.

Exact ordinary source f7 is `f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d`. Exact source98 is
`98eac12dc2595a8742a81b58e96b899fed905194`. The f7→source98 engine difference is the explicit
deterministic-collective policy requirement; the raw/group difference within source98 is grouped
prefill. Source98 worker startup receipts bind all four actual PIDs to HCCL=true, batch invariance
unset, the requested group role and native/isolated TP/EP initialization. These are observed
environment and initialization controls, not proof of every collective algorithm's determinism.

## Installed Manager source bridge

Both source98 Manager runs use the installed Manager with an explicit source path and
`--deterministic-collectives`. Each independently verifies 22 prepared snapshot hashes,21
source-runtime hashes, the installed manifest/Manager identity, reviewed configuration and source
import origin. Parent HCCL starts unset; all four worker readbacks show HCCL=true and the requested
raw/group role. The journal binds the launch to four worker components plus scheduler and admission
state. The runtime state changes inactive→effective→inactive. Each API call returns HTTP 200 for the
fixed 128-token prompt and reports eight completion tokens.

The API response has `token_ids=null`; this is serving continuity, not a Manager token-equivalence
result. Raw and group Manager presets use their respective compiler backends, so their serving
results are not a matched numerical or performance comparison. The deployment is source-based; no
new wheel is qualified by these runs.

Both intentionally stopped Manager processes retain actual exit 130 in separately typed capsules.
Controller success means the prescribed serving/startup/stop checks passed; it never rewrites exit
130 to 0. The outer controllers sent only the planned SIGINT and recorded no surviving owned
processes, no forced outer cleanup and empty physical devices. Native logs nevertheless retain abort
mode, timeout 0, a process-manager force-kill count 1, EngineCore SIGTERM, and resource_tracker
warnings for four semaphores and six shared-memory objects. The source98 plain workers also log
native SIGTERM fallback after their grace period. This is bounded cleanup with those fallback paths,
not signal-free or globally graceful shutdown. The resource_tracker warnings alone do not prove
resources remained after termination.

## Single native quality control and preserved budget failures

The original 2048-token native attempt was deliberately stopped after truncation made its strict
completion criterion unattainable. Its terminal result contains 39/200 outputs: 36 stop and 3
length; the collection is incomplete and remains failed. Earlier stop-decision/calibration snapshots
retain their own timestamps, counts and hashes. They are not substituted for the final 39-output
record.

The separately frozen 8192-token attempt completed all 200 native responses: 191 stop and 9 length.
The unchanged scorer produces 165/200 correct (82.5%); all nine length responses are wrong. The
original strict all-complete condition is false. The predeclared budgeted supplement accepts these
terminal response types for collection, while retaining truncation-as-wrong. Its existence does not
repair strict v1/v2 failure.

The single-control validation rechecks frozen gold/parser scoring, input and admission binding,
terminal custody, and token/text binding. This is descriptive control 1 only: there is no second
fresh native control or candidate quality arm in this evidence selection, so no
stable-both-native-correct veto, candidate floor, paired accuracy comparison or supplementary
acceptance is evaluated. The 23:25 supplementary freeze snapshot from v2 remains a separate
historical snapshot; its partial counts must not be mixed with either terminal run.

## Actual pressure passed; cancellation coverage failed

`raw-pressure-suite-hccl-true-01` remains overall FAIL. Its first phase completed three outputs in
129 steps, observed 64 allocation-None events and one seed-D preemption while pending P retained 15
physical blocks. The phase returned all 19 usable blocks and no live request, cohort, held ownership
or ACK before the next phase. Its final audit passed after shared model shutdown.

The partial-cancellation phase failed after six steps because the required original two-peer
pending-P8 boundary never occurred. No abort was issued; there is no cancellation/ACK/reuse success
to report. The full-cancellation phase was not admitted. Complete original traces and both
provisional/final audits remain preserved.

The independent source audit and exact-method CPU replay explain the missed boundary. After seed 4
blocks plus two peer 7-block allocations, one block was free. Pinned Ascend's 0.25.1 Mamba
allocation predictor interpreted the running request's computed tokens as external and added one
estimated block for each of three Mamba groups, after the native running fast path needed zero. Its
capacity check rejected 3>1, the seed was preempted, and with no D remaining the plugin completed
pending P4→P40. The required P8 cancellation point disappeared. Those extra blocks are an allocation
estimate, not observed temporary allocation or copy-on-write. The original physical occupancy
estimate was insufficient to guarantee the intended cancellation boundary. No native runtime fix is
part of this evidence batch.

The subsequently prepared 22-block cancellation treatment is a separate experiment, preserving
original prompts and cancel/ACK/reuse conditions. All of its device attempts fall after the fixed v3
evidence cutoff and are excluded. No future result may be retroactively inserted into the failed
20-block suite.

## Investigation disposition and remaining gates

For #11, the retained evidence supports a bounded, configuration-specific disposition: on the frozen
discriminator inputs, same-card HCCL=true controls in the prior v2 source6c study show matched
observed vectors where comparable; ordinary f7/source98 workers preserve the fixed API outputs.
Source98 makes that environment requirement explicit and exercises it through the installed Manager.
This is a reproducible mitigation/qualification boundary. It does not identify the universal cause
of the historical divergence, retroactively invalidate its failures, prove arbitrary-input
equivalence, qualify a wheel or establish all backend variants. Closing that investigation, if
chosen, must preserve these limits and keep the independent quality and performance gates open.

#23 remains open for the two-native-control and raw/group quality comparison, including the
stable-correct veto and all paired transitions. #3 and draft PR #10 still require current matched
ordinary-worker performance measurements and remaining lifecycle coverage. Prepared no-profiler ABBA
and separate mixed-P/D profile tools are plans, not results. This batch provides no current latency,
throughput, goodput or group-FX speedup claim, and does not replace historical performance evidence
with numbers from different versions.

## Custody and publication status

V3 selects nine newly terminal run capsules and one explicit support capsule. The Manager capsules
use a distinct terminal schema that preserves intentional exit 130 and the full
installed/runtime/cleanup bindings. The older generic capsule builder is unchanged. Seven
ordinary/quality/pressure capsules retain their already frozen identities; none are rebuilt or
relabeled.

An early Manager validator imported a helper and generated two new bytecode cache files after the
runs ended. The side-effect receipt preserves those new bytes outside the run directories and
records their approved removal; existing original files were unchanged. The final validator uses a
no-write loader, and full inventories match before and after validation and archive construction.
This history is disclosed rather than claiming the directories were never touched after termination.
The new cache artifacts are support evidence and are not executed during archive verification.

The support capsule contains the six comparison originals and comparator, single-native
validation/scorers, source98 bridge reviews and producers, pressure diagnosis and CPU reproduction,
both capsule builders and their validation, this report, exact terminal selection, and the source98
committed tree export with nested member identities. It contains no model weights or Git history.
Explicit selection and file hashes define its scope.

All archives are locally byte/member verified. Planned Release locations are publication candidates
only; v3 upload and independent public readback have not occurred at preparation time. The existing
21-asset v2 readback and later two-asset quality-protocol readback remain unchanged historical
snapshots. New remote receipts must append their own version. No active experiment, website data,
official baseline or leaderboard entry is included in this update.
