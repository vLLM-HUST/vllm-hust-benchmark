# MMLU-Pro 12,032 paired analysis (single-card TP1)

| arm | correct/n | accuracy | 95% Wilson |
|---|---:|---:|---|
| bf16 | 9609/12032 | 79.862% | [79.14, 80.57] |
| w8a8 | 9565/12032 | 79.496% | [78.77, 80.21] |
| w4a8_gcy | 9595/12032 | 79.746% | [79.02, 80.45] |
| w8a8nr | 9526/12032 | 79.172% | [78.44, 79.89] |

## Paired McNemar (exact)

| comparison | n | only A correct | only B correct | delta (A-B) | Wald 95% | p |
|---|---:|---:|---:|---:|---|---:|
| w8a8nr vs bf16 | 12032 | 440 | 523 | -0.690 pp | [-1.195, -0.184] | 0.008198 |
| w8a8nr vs w8a8 | 12032 | 500 | 539 | -0.324 pp | [-0.849, +0.201] | 0.2384 |
| w8a8nr vs w4a8_gcy | 12032 | 449 | 518 | -0.573 pp | [-1.080, -0.067] | 0.02871 |
| w8a8 vs bf16 | 12032 | 420 | 464 | -0.366 pp | [-0.850, +0.119] | 0.1481 |
| w4a8_gcy vs bf16 | 12032 | 456 | 470 | -0.116 pp | [-0.612, +0.379] | 0.6693 |
| w4a8_gcy vs w8a8 | 12032 | 498 | 468 | +0.249 pp | [-0.257, +0.756] | 0.3508 |

Protocol: frozen 12,032-case MMLU-Pro plan; /v1/completions; max_tokens=2048; temperature=0; seed=17; concurrency=16; no retries; 32K context; no APC; no MTP.
BF16 is the published TP2 baseline; W8A8 / W4A8(gcy) / W8A8NR are TP1 single-card. Cross-topology BF16 comparisons carry that caveat.
