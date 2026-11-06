# PyramidKV Qwen3.5 long-109: 0.4.0rc2.dev0 functional acceptance

**One fresh ON lifecycle on 2026-11-06; TP2/C128 Eager diagnostic, excluded from official
leaderboards. Functional acceptance passed. No paired performance or task-quality claim.**

The candidate fixes resumed prefill crossing the original prompt boundary before the first
compression. In this run, `long-109` actually reached 6144 computed tokens, with native scheduling
of 1025 tokens capped to 1024; its 7168-token prompt then committed compression to 2048 physical
tokens, acknowledged once by each TP rank. This is direct evidence for the new run. The historical
November 4 logs lacked that request-level scheduling trace and do not establish the old run's cause.

| Acceptance gate                      | Observed result                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| Completed requests                   | 178: 48 short + 128 long + 2 readiness                                                |
| Target compression                   | Exact `long-00` through `long-127`, 128/128 including `long-109`                      |
| Scheduler / TP-rank acknowledgements | 129 / 258, including long readiness                                                   |
| Long workload                        | 7168 input / 512 output tokens                                                        |
| Server concurrency                   | Running + waiting reached 128 before the first completion                             |
| Native pool at startup and final     | 558 blocks / 2048 logical page size / 457,113 effective slots                         |
| Native maximum-length concurrency    | 55.8                                                                                  |
| Final serving state                  | Zero running and waiting requests; no ERROR or Traceback                              |
| Shutdown / hardware release          | Return code 0; physical 4/6 report no running processes                               |
| Source tests                         | 303 passed, 5 skipped; 4 NPU tests and 1 host plugin bootstrap test separately passed |

The native pool is 409 slots above the historical requested 456,704 slots. Hybrid capacity is not
physical blocks multiplied by logical page size. Devices are two Ascend910B2 64 GiB cards, physical
**4/6**, unlike historical physical 1/5. These observations establish one functional ON acceptance,
not OFF/ON speedup, repeatability, or answer quality. Raw timing and aggregate files remain
available for audit, without promoting them to a comparison.

## Candidate, source and publication lineage

The accepted candidate source is
[`77b5c7fd8817b14c9800f6369bd4db622f53f43e`](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/commit/77b5c7fd8817b14c9800f6369bd4db622f53f43e).
The wheel `vllm_ascend_pyramidkv_hust-0.4.0rc2.dev0-py3-none-any.whl` has SHA256
`5392b2a8afa971cf0af1b9928d693ab80a71976256de87e7f85b5f9c3a4819be`. The final sdist SHA256 is
`0463ea98ffbc2082615f4db357956185dd95c9c6710d5b3a5935e4eedf2bc931`. The capsule includes selected
committed source files and raw Git commit/tree objects; the verifier recomputes Git object
identities, resolves each source path from that commit, and compares every package file against both
the wheel and the executed candidate overlay.

The published public development candidate release is
[`v0.4.0rc2.dev0`](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/releases/tag/v0.4.0rc2.dev0).
Publication, merge, and public download acceptance are separate delivery actions. The original
`acceptance.json` and original README correctly retain their capture-time **not published** status.
The separately checksummed `delivery-receipt.json` records subsequent merged-source identity, public
artifacts, and readback; it does not change the original run. The offline verifier checks the exact
retained receipt bytes, merged Git commit object, successful source PR/main check results, captured
public prerelease API, both public artifact identities and clean installation receipts. It does not
check current network availability. The merge is `3db81efc6a3ff9c7e1a5f63d9b890e20e4f687c6`, with
the same source tree `05313664b8cc37553eecb6f6aa0140bdcf93aceb` as the accepted commit.

Pinned host source commits are vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804` and vLLM-Ascend
`03766ac696fde5ab1980d80ca0b8543d3580c989`. The run uses BF16, TP2/PP1, Eager/no MTP, 8192 context,
batch-token budget 4096, seed 0, prefix caching, chunked prefill, async scheduling, Mamba cache
align and 128 maximum sequences. The frozen client uses an unlimited connector so it does not cap
the target concurrency. PyramidKV settings remain capacity 512, beta 20, window 8, kernel 7,
maxpool, KV-head/mean GQA, no merge and minimum prompt 4096. Full request contents, launchers,
settings and original logs are archived.

## Preserved contract and historical failure

`contract.json` is preserved byte-for-byte, including its inherited stale descriptive fields:
`candidate_source_commit` names the prior candidate `a822bf7...`; `aggregation` still says one
OFF/ON pair; `hardware_change` and several `limitations` still describe physical 1/5 and the prior
unchanged-wheel comparison. Those fields describe inherited history, not this run. The contract's
explicit `scope`, `physical_devices: [4,6]` and single enabled `lifecycles` entry, together with the
new raw evidence and accepted source identity above, define this functional rerun. No field was
silently rewritten. Every `files_sha256` entry still matches. Its inherited `prior_contract_sha256`
still points to the November 3 pressure `contract-v2.json` (SHA256
`03a2592aca7ca9de81979824383832660ad49c8aa636c62967af5428a95c0c13`), not the immediate November 4
contract. Both ancestor contracts are included; the verifier checks that inherited pointer against
the November 3 bytes and its November 4 predecessor. The immediate November 4 contract SHA256 is
`e6f69e5485f11035abf5bd165dc319b7c6b004989a09460ff5159eb9e9179365`; the preserved new contract
SHA256 is `90546cd23a8d2a81886b4b0a454d13f108978fab1a6315e0ad263ddb94e51d6e`.

The [historical pool report](../pyramidkv-qwen35-pool456704-20261104/README.md) remains a **failed
full-compression criterion**: 127/128 target requests plus readiness, 128 scheduler commits / 256
ACKs; `long-109` completed but did not compress. This capsule preserves its original server log,
frozen contract, analysis amendment and failed analyzer log under `supplemental/historical/`. The
current strict source audit must still reject that log for missing `long-109`. The separate
[rc1 capsule](../pyramidkv-standalone-rc1-20261103/README.md) retains its own earlier acceptance
boundary.

Original early red tests, independent review, rejected historical audit, original and final source
patches, pre-final/final artifacts, tests, clean-install results and Manager lifecycle receipts are
retained. The copied original `audit.py` is unchanged; the strict current audit is separately
included at `supplemental/source/scripts/model-comparison/audit.py`.

## Offline verification and archive format

With Python 3.11 or newer, CPU only, no model/runtime packages and no network:

```sh
python verify.py
python test_verify.py
```

`SHA256SUMS` inventories public files; deterministic `members.json.gz` records each archived
member's path, archive, uncompressed byte length and SHA256. Archives use normalized tar metadata
and gzip timestamp zero, and each is below 50 MiB. All 617 original checksummed files and the
original checksum manifest are retained; installed environment trees `clean-install/`, `manager/`
and Python caches were outside that original checksum inventory and are excluded. Their original
logs and lifecycle results remain included.

The verifier refuses unsafe paths, links, duplicate, unknown or missing members, mismatched
lengths/hashes and unexpected public files. It checks every original and contract hash, reuses the
exact original `verify_streams.py` and strict committed source audit, and adds independent checks of
request/response identities, full SSE/event agreement, usage, text, all per-request timing metrics,
group results/aggregates, telemetry against raw metric text, actual server admission before
completion, native startup/final pool, zero final state, clean shutdown and explicit device-release
receipts. It also verifies source/wheel/overlay identity and actively rejects the historical
incomplete run. Hashes bind these retained observations; they do not independently establish remote
hardware truth or replay inference.

The retained tests show 303 generic passes and five skips. The separate NPU/host logs cover those
opted-in cases. This CPU verifier does not rerun those hardware tests, validate model weights
against their filesystem stat pins, or make new runtime or publication claims.
