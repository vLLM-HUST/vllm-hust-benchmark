# 旧生成器处置

28项旧实现已从活动包移除，不提供synthetic fallback。旧版见commit a5604688；本机另保存重构前含未提交修改的完整快照。删除的是构造代码和专用测试，不是数据或历史结果。

| 旧ID                                    | 原分类决定                  | 新分类/维度                                                                         | 实现                     |
| --------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------- | ------------------------ |
| session-affine-multi-turn               | CANONICALIZED               | conversation-continuation                                                           | 已移除；显式选择来源适配 |
| session-affine-bursty                   | MERGED                      | conversation-continuation、bursty-arrival                                           | 已移除；显式选择来源适配 |
| rag-followup                            | RETAINED                    | rag-followup                                                                        | 已移除；显式选择来源适配 |
| long-context-doc-analysis               | CANONICALIZED               | document-grounded-analysis、long-context                                            | 已移除；显式选择来源适配 |
| tool-scaffold-agent                     | SPLIT_DIMENSIONS            | tool-interaction、tool-schema-reuse                                                 | 已移除；显式选择来源适配 |
| repo-aware-coding-assistant             | CANONICALIZED               | interactive-repository-assistance                                                   | 已移除；显式选择来源适配 |
| experiment-planning-assistant           | RETAINED_CANDIDATE          | scientific-experiment-planning                                                      | 已移除；显式选择来源适配 |
| simulation-analysis-verification        | RETAINED_CANDIDATE          | simulation-analysis-verification                                                    | 已移除；显式选择来源适配 |
| async-document-pipeline                 | PROFILE_ONLY                | document-grounded-analysis、structured-generation、text-transformation、async-batch | 已移除；显式选择来源适配 |
| code-eval-judge                         | CANONICALIZED               | code-evaluation                                                                     | 已移除；显式选择来源适配 |
| structured-json-generation              | CANONICALIZED               | structured-generation                                                               | 已移除；显式选择来源适配 |
| multi-turn-support-chat                 | MERGED                      | conversation-continuation、customer-support                                         | 已移除；显式选择来源适配 |
| realtime-voice-assistant                | RETIRE_PROXY_AS_REAL_FAMILY | text-carried-media-proxy、audio                                                     | 已移除；显式选择来源适配 |
| session-continuation-maintenance        | MERGED                      | conversation-continuation、maintenance-resume                                       | 已移除；显式选择来源适配 |
| shared-prefix-multi-tenant-assistant    | PROFILE_ONLY                | exact-prefix-sharing、multi-tenant-soak                                             | 已移除；显式选择来源适配 |
| dynamic-rag-corpus-update               | CANONICALIZED_CANDIDATE     | evolving-corpus-rag                                                                 | 已移除；显式选择来源适配 |
| memory-write-then-reuse                 | CANONICALIZED_CANDIDATE     | memory-write-read                                                                   | 已移除；显式选择来源适配 |
| preemption-resume-long-decode           | PROFILE_ONLY                | preemption-resume                                                                   | 已移除；显式选择来源适配 |
| mixed-prefill-decode-interference       | PROFILE_ONLY                | prefill-decode-mix                                                                  | 已移除；显式选择来源适配 |
| overload-backpressure                   | PROFILE_ONLY                | overload-recovery                                                                   | 已移除；显式选择来源适配 |
| cancellation-timeout-churn              | PROFILE_ONLY                | cancel-timeout-retry                                                                | 已移除；显式选择来源适配 |
| kv-eviction-recompute                   | PROFILE_ONLY                | working-set-revisit、exact-prefix-sharing                                           | 已移除；显式选择来源适配 |
| multi-tenant-fairness-soak              | PROFILE_ONLY                | multi-tenant-soak                                                                   | 已移除；显式选择来源适配 |
| multimodal-online-serving               | CANONICALIZED               | multimodal-understanding、image                                                     | 已移除；显式选择来源适配 |
| scientific-code-agent                   | CANONICALIZED               | scientific-coding-workflow                                                          | 已移除；显式选择来源适配 |
| semantic-similarity-routing             | PROFILE_ONLY                | semantic-near-duplicate                                                             | 已移除；显式选择来源适配 |
| prefill-decode-disaggregation           | PROFILE_ONLY                | pd-disaggregation                                                                   | 已移除；显式选择来源适配 |
| long-horizon-software-engineering-agent | CANONICALIZED               | repository-agent-loop                                                               | 已移除；显式选择来源适配 |
