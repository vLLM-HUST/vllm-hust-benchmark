# Engine Release History

`data/engine-release-history-v1.json` maps frozen B1 r4–r8 releases to their own
configuration, MODs and scalar observations. It is the canonical source for the
website release menu; the website must verify its publication hash before display.

- r4: original 2026-10-30 B0/r4 MMLU200 and C8 pairs, projected from the unchanged
  public `reports/full-answer-joint-20261030` summaries.
- r5: failed before service readiness; the r6 overlay explicitly records the fix.
  No r5 formal inference values are present.
- r6: the existing five-profile report remains the complete source and renderer.
- r7: one historical r6 run versus one r7 run at C8, with the original control
  identity retained. The 138/200 MMLU result and original 144/200 internal screen
  remain separate from this short-output performance comparison. P95/P99 values
  remain available, including the higher r7 P95 TTFT.
- r8: the existing new B0/r8 report remains the current source and renderer.

Source URLs use fixed commits; SHA256 and byte lengths identify the original
records. Engine source links require repository access. This projection contains
no prompts, gold labels or full model responses. Historical protocols and their
failure records are retained; this update launches no inference and rewrites no
sealed evidence.
