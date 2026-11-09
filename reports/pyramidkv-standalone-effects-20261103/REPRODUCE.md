# Offline verification and input reconstruction

Use Python 3.10 or newer. Public verification uses only the standard library and makes no network,
model or NPU calls:

```bash
python reports/pyramidkv-standalone-effects-20261103/verify.py
python -O reports/pyramidkv-standalone-effects-20261103/verify.py
```

Run the delivered verifier from an immutable benchmark commit. It checks every top-level report file
against `SHA256SUMS`, reads the exact regular archive-member inventory, and verifies the originals
without extracting or executing the archived scripts. Large member inventories use
`members.json.gz`; small inventories use `members.json`. Raw archive parts are concatenated in their
numbered order. Both inventories and archive parts are covered by the outer checksums.

The verifier joins frozen contracts, launch/runtime receipts, request membership, SSE responses,
synthetic request payloads, compression commits and both TP acknowledgements. It checks recorded
group aggregates, engine/HTTP counts, telemetry boundaries and the pressure-cache evidence, then
recomputes `analysis.json`. A checksum match alone is not the acceptance test. Client latency values
remain observations made by the original client: the retained SSE bodies have no per-token receive
timestamps and cannot independently reconstruct TTFT or TPOT.

## Private input coverage

Public verification reports zero private payloads verified and zero official F1 scores recomputed.
It checks the saved score aggregation, public predictions and public evidence. It cannot reproduce
the omitted questions, references, tokenized QA prompts or complete original private case plans.

For dataset-backed verification, obtain the official LongBench files through the applicable access
and licensing terms and keep them in a private directory:

```text
PRIVATE_ROOT/
  data/narrativeqa.jsonl
  data/qasper.jsonl
  data/2wikimqa.jsonl
  quality-plan/cases.json
```

The official dataset is `zai-org/LongBench` (formerly `THUDM/LongBench`), revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`. Use the v1 source/templates at LongBench commit
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`, with its original `LongBench/config/` directory. The
three JSONL file hashes and prompt-template hash are recorded in the frozen quality manifest.

Rebuild the private case plan with the tokenizer/model files from Qwen3.5-35B-A3B revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`. Use the evaluation runner from PyramidKV source
`06ab70a5dc58b6a2d4f1995c9d68ac61216d5b6a` (the tested tree merged through PR #34); the exact runner
bytes are also retained in the capsule and pinned by both contracts. The serving plugin itself is
the published `0.4.0rc1` wheel, source `bdc20e248c9202560bf44c086bcdede2952b3274`.

In an environment containing the runner's tokenizer dependencies:

```bash
python scripts/qwen35/evaluate.py prepare \
  --model-path /path/to/Qwen3.5-35B-A3B \
  --longbench-repo /path/to/LongBench \
  --data /private/reproduce/data \
  --output /private/reproduce/quality-plan \
  --samples-per-task 100 \
  --capacity-concurrency 1 2 4

python /path/to/benchmark/reports/pyramidkv-standalone-effects-20261103/verify.py \
  --private-root /private/reproduce
```

The preparation command runs in the pinned PyramidKV checkout. Its output must reproduce the
original quality-plan hash exactly. Verification reconstructs all three private plans from that
quality plan and the public synthetic cases, then rebuilds all 696 QA request payloads and checks
their original byte lengths/hashes. It recomputes F1 against the official references and checks row
identities. Private request files from the measurement machine are not required.

Do not publish the private root. `private-inputs.json` records hashes and byte lengths for the three
case plans, three official data files and 696 omitted request payloads. It does not contain their
contents. Keep verification output outside the report directory so an extra file does not violate
the exact top-level inventory.

## Re-running measurements

Offline recomputation does not claim a second live measurement. A new live series needs a separately
allocated two-device window, the pinned host/native operator identities, restored model files and
the complete frozen inputs. Review the retained launcher and lifecycle driver before adapting local
paths. Preserve both reservation budgets, OFF/ON/ON/OFF ordering, warmups, probes, fixed output
lengths, capacity operating points and stop rules. Record a new experiment identity rather than
replacing these originals.

`urlopen(timeout=600)` is a socket connect/read timeout. Whole-client time limits are 7200 seconds
for the quality lifecycles and 3600 seconds otherwise, within the preregistered eight-hour window.
These settings do not establish a 600-second streaming-request deadline or a production latency SLO.
