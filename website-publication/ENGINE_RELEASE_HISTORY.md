# Engine Release History

`data/engine-release-history-v1.json` maps frozen B1 r4–r8 releases to their own configuration, MODs
and scalar observations. It is the canonical source for the website release menu; the website must
verify its publication hash before display.

- Early r1–r3: packed-host attempts preceding frozen r4, kept separately from frozen Engine
  releases. Retain the memory-guard interruption, pre-model port failure and partial 8GiB/rank
  pinned-allocation failure. No complete-score charts.
- r4: original 2026-10-30 B0/r4 MMLU200 and C8 pairs, projected from the unchanged public
  `reports/full-answer-joint-20261030` summaries.
- r5: failed before service readiness; the r6 overlay explicitly records the fix. No r5 formal
  inference values are present.
- r6: the existing five-profile report remains the complete source and renderer.
- r7: compare against the sealed fixed B0 C8 observations from the r6 report. The B0 seal and reuse
  identity are explicit. The original r6-to-r7 development comparison remains separately under
  `development_measurements`, with quality and tail values retained.
- r8: all five fixed performance pairs completed on 2026-11-18, with 672 requests per arm and zero
  failures. The catalog includes the scalar observations from
  [the audited r8 report](../reports/r8-fixed-five-20261118/README.md); existing 4K/16K continuous
  observations remain additional rows. Same-model grading and unresolved cases are disclosed
  separately; performance completeness does not assert complete project acceptance.

Source URLs use fixed commits; SHA256 and byte lengths identify the original records. Engine source
links require repository access. This projection contains no prompts, gold labels or full model
responses. Historical protocols and their failure records are retained; this update launches no
inference and rewrites no sealed evidence.

The public visibility policy permits only this catalog’s exact SHA256-bound bytes. Changing a
measured value and re-signing the source manifest still fails publication; unrelated withdrawn
research records retain their existing filters.

All primary version comparisons use the fixed B0 vLLM0.18 / Ascend0.18 / CANN8.5.1 identity, image
and SDK source-map hash. Each observation keeps its original measurement values, warmup protocol and
seal. A B1 release is never used as the primary denominator.

## r8 上下文与四租户补充

`r8.context_tenant_supplement` 保存本轮 original018 / 冻结 r8 的真实4K、16K、32K和四租户配对、全部窗口与租户分布；原五项
`measurements` 字节内容保持原来源。输入为五份完整公开 SWE / Terminal
历史、C8/四租户C24、out256，正式提交300秒后排空；它不替代完整回答预算，也不产生有效算力效率。

[报告与原件](../reports/r8-context-tenant-20261119/README.md)和 `data/r8-context-tenant-20261119/`
是新增来源。网站消费更新后的固定 benchmark 提交及目录摘要，所有版本继续保留缺项、固定五项与相同补测位置。

## r8 有效算力参考（2026-11-19）

r8.compute_reference
接入已封存的固定MMLU200/out256三轮完整参考分母复用认证。B1下界89.7838%，50%里程碑达成、90%最终目标未达成；不消去不同F。[统计、原件与复算](../reports/r8-compute-reference-20261119-v2/README.md)。
