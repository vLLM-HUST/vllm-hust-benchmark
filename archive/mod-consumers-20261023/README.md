# Four MOD consumer migration capsules

Continues [#326](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/326). These are
byte-preserved subsets of already public Git history, not new experiments or acceptance. Original
run IDs, failures, diagnostic labels and context are retained. Private execution-node records
omitted by the original Profiler exporter remain omitted here.

| Capsule / manifest                                   | Selected source directory                                                       | Files including provenance context | Consumers to migrate                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------: | ---------------------------------------------------------------------------- |
| [KV Arrival](kv-arrival/manifest.json)               | `paper/kv_materialization_control/experiments/results/live/`                    |                                 22 | README/historical read entry; experiment output paths remain output paths    |
| [PegaFlow](pegaflow/manifest.json)                   | `results/issue23/20261015-contention/`                                          |                                 58 | contention verifier, corruption regression, rendering input and report links |
| [MuxPD](muxpd/manifest.json)                         | `evidence/2026-10-10/`                                                          |                                 41 | TP2 evidence links; source/runtime tests do not read this subset             |
| [Request Lifecycle Profiler](profiler/manifest.json) | `.benchmarks/results/m0_issue19_public/20260822-sanitized-corrected-paired-10/` |                                275 | public-export verification test, Make target and claim references            |

The four archives are
[Release attachments](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-mod-consumers-20261023-v1).
Each manifest fixes the source repository/commit, archive size/SHA256 and every member's original
path, Git blob, byte length and SHA256. Context files are snapshots, not replacements for current
MOD code. A file count is not a run count. The Profiler's `NO_GO` and MuxPD's eager diagnostic scope
do not become successful public-target benchmarks after migration.

Read without executing archived scripts:

```bash
python scripts/fetch_mod_consumer_evidence.py \
  --manifest archive/mod-consumers-20261023/pegaflow/manifest.json \
  --destination /absolute/new/external/evidence-directory
```

The downloader rejects checksum/length mismatches, missing/duplicate members, links and unsafe
paths. It never falls back to local MOD evidence. Source cleanup requires actual remote readback and
member verification plus consumer tests; source PR/merge receipts are recorded in #326. The 396
archived files include context retained in the MODs; not all are deletion targets.

Remaining scope: other dates/directories and mixed implementation/research documents in these four
MODs, all other MOD queues, installation consumers and external hardware/review gates remain open.
No old Release installer is deleted, lost probe searched or cancelled GDN test resumed.

Skill review: E01/E04/E31/E37/E42/E43/E45. Current request is consumer migration; verify fixed
source bytes, preserve failed/negative outcomes, repair code readers instead of skipping tests,
regenerate both navigation indexes, and merge only applicable green CI. No hardware execution.
