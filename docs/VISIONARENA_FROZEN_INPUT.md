# VisionArena offline frozen input

The official VisionArena workload does not require the full 84 GB dataset on an
offline runner.  With `datasets==3.3.0`, streaming enabled, seed `0`, the default
1,000-row shuffle buffer, and 1,000 requested prompts, canonical shard 40 is
shuffled first and supplies every selected request.

Prepare and verify the local one-shard view:

```bash
python scripts/prepare_visionarena_frozen_input.py
```

The command verifies the pinned Hub tree, the shard size and SHA-256, the exact
source-row selection, and a content digest covering each selected conversation
ID, prompt, and original image bytes.  It then creates a symlink-only local
dataset at the path below. Run it with the same Python environment that will
launch the benchmark; the generated adjacent runtime receipt records the exact
installed `datasets` version after loading and hashing all 1,000 requests.
The content digest has been reproduced in offline mode with both the reference
`datasets==3.3.0` loader and the locally installed `datasets==5.0.1` loader.

```text
/data/shared_datasets/vllm-hust-benchmark/huggingface/frozen-inputs/VisionArena-Chat-1394b4f-seed0-1000
```

Run the official v0.18.0 client offline by replacing only its physical dataset
location while retaining the logical Hugging Face name. For the reference
v0.18.0 environment, keep `datasets==3.3.0` as recorded in the manifest:

```bash
python -c 'import datasets; assert datasets.__version__ == "3.3.0"' && \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m vllm.entrypoints.cli.main bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --host 127.0.0.1 \
  --port 8010 \
  --model /data/shared_models/Qwen--Qwen2.5-VL-7B-Instruct \
  --dataset-name hf \
  --dataset-path /data/shared_datasets/vllm-hust-benchmark/huggingface/frozen-inputs/VisionArena-Chat-1394b4f-seed0-1000 \
  --hf-name lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --seed 0 \
  --num-prompts 1000 \
  --request-rate 1
```

The local directory contains exactly one Parquet data file.  Its first 1,000
shuffled outputs are therefore the same canonical requests that the pinned
43-shard streaming dataset produces.  The local path is physical provenance;
the logical workload remains `lmarena-ai/VisionArena-Chat` at revision
`1394b4f59ab6f1f2e5aff6bc15b448e15960e170`.

The same frozen input is reusable for the three currently relevant milestones;
only the checked-out engine/plugin pair changes:

| Milestone | vLLM commit | Ascend plugin commit |
| --- | --- | --- |
| Official v0.18.0 | `bcf2be96120005e9aea171927f85055a6a5c0cf6` | `e18643f8a4d5bd9990727654318ad069ea0b56e2` |
| Pre-jump historical point | `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7` | `312ca80a90cbd28438bce3b59e3fbaad749451f3` |
| Post-jump historical point | `ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` | `312ca80a90cbd28438bce3b59e3fbaad749451f3` |

For every run, retain the adjacent runtime receipt and record the actual
`datasets` package version in the experiment artifact. Hostname, IP address,
and rack remain provenance only; the comparable hardware identity is 1×910B2.

The committed audit contract is
[`visionarena-chat-1394b4f-seed0-1000.manifest.json`](dataset-manifests/visionarena-chat-1394b4f-seed0-1000.manifest.json).
