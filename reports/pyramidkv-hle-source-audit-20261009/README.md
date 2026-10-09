# HLE-Verified source and modality audit

This input-only audit advances
[PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8) and
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). It makes no
solver/judge calls, freezes no execution subset, and reports no task scores or optimization benefit.
The canonical rights-review gate is still open.

The frozen revision `b705e0fb541c025a1532ce0d60d70ae2f53b00e0` belongs to
[the GitHub source repository](https://github.com/SKYLENAGE-AI/HLE-Verified/tree/b705e0fb541c025a1532ce0d60d70ae2f53b00e0),
not the similarly named Hugging Face repository. Using it as a Hugging Face revision returns 404.
The three GitHub JSONL files were downloaded locally and their complete bytes verified against that
commit's Git LFS SHA-256 pointers. This is not a new claim to have repeated the other host's full
1,135-file audit.

| Source subset | Tasks | Problem images | Answers represented as non-string scalars |
| ------------- | ----: | -------------: | ----------------------------------------: |
| Gold          |   668 |             93 |                                         0 |
| Revision      | 1,143 |            118 |                                       144 |
| Uncertain     |   689 |            131 |                                         0 |

All 2,500 IDs are unique across the three source subsets. The 144 non-string Revision answers are
109 integers, 21 booleans and 14 floats. The inventory preserves their types and hashes their
canonical JSON values; converting every value to a string, or treating `False`/zero as a missing
answer, would change the source contract. Each row also records its source line, question and answer
hashes, image-field hashes and modality requirements. Question/answer contents and source images are
not republished.

The image counts use only problem images and previews. Rationale images belong to reference
explanations and must not enter solver inputs. No preview-only task was found in this snapshot. The
current qualified Qwen/PyramidKV launch sets image and video limits to zero. In particular, removing
the 93 image-bearing Gold tasks would leave 575 tasks and **would not be the complete 668-task Gold
evaluation**. An image-aware runtime and scorer protocol must be validated first.

The pinned repository tree has no LICENSE/COPYING file, and the inspected README has no explicit
license grant. This records missing provenance; it is not a legal conclusion about all possible
permissions. The organization-level rights gate must be resolved before choosing/freezing the formal
execution subset and publishing task scores. Source metadata auditing alone does not satisfy that
gate or select a judge.

The two local inventory generations were byte-identical. `summary.json` records each uncompressed
inventory hash; `SHA256SUMS` covers the published gzip archives and provenance receipts. The gzip
archives contain metadata only. The source-tree response and source-retrieval record are included;
the README Git blob hash and local script hash are in `audit-receipt.json`.

Reproduce from the pinned GitHub/LFS files, plus `README.md` and the recursive tree API response
saved as `tree.json` in the asset directory:

```bash
python scripts/audit_pyramidkv_hle.py \
  --assets /path/to/pinned-hle-verified \
  --output /path/to/new-empty-audit
```

Do not substitute current Hugging Face Parquet files without independently establishing and
recording their correspondence to this frozen GitHub snapshot. Do not treat source `answer_type`
labels as a complete scorer specification: judge identity, prompts, calibration, image handling,
budgets and failure policy remain to be frozen after rights review.
