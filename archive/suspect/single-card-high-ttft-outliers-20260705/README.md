# Single-card high-TTFT outliers archived on 2026-07-05

These submissions were removed from public leaderboard inputs because same-spec NPU0 reruns showed
the original single-card online rows were high-TTFT outliers rather than stable performance
measurements.

The sharegpt-online replacements and all four instructcoder-online replacements now have
zero-failure NPU0 reruns imported. The first ascend-pr66-simllm-kv-manager instructcoder-online
retry failed 242/2048 requests (error_rate=0.1181640625) and was not imported; a second retry
completed successfully and replaced the archived high-TTFT row.

| Old submission                                                                                      | Replacement                                                                                                  | Status                      |     Old throughput |           Old TTFT |     New throughput |           New TTFT |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------- | -----------------: | -----------------: | -----------------: | -----------------: |
| `historical-pr-pr77-perfgate-l2-scenario-registry-sharegpt-online-ceec19abb0-51e577b17b`            | `historical-pr-pr77-perfgate-l2-scenario-registry-ttft-fix-sharegpt-online-ceec19abb0-51e577b17b`            | imported                    | 142.98654890992057 |  22094.54369746265 | 191.56842611430145 |  199.9608273850754 |
| `historical-pr-pr69-perfgate-l2-targeted-verification-sharegpt-online-ec4847981f-51e577b17b`        | `historical-pr-pr69-perfgate-l2-targeted-verification-ttft-fix-sharegpt-online-ec4847981f-51e577b17b`        | imported                    | 137.54830155261115 |  28465.51688949112 | 191.43809178404737 | 198.76766856992617 |
| `historical-pr-align-ascend-benchmark-perfgate-defaults-sharegpt-online-2fb7859dd0-51e577b17b`      | `historical-pr-align-ascend-benchmark-perfgate-defaults-ttft-fix-sharegpt-online-2fb7859dd0-51e577b17b`      | imported                    | 156.29460906068576 | 15664.183726105839 | 191.75260198702145 | 198.51446812739596 |
| `historical-pr-align-perfgate-baseline-spec-source-sharegpt-online-dcc06b18f3-51e577b17b`           | `historical-pr-align-perfgate-baseline-spec-source-ttft-fix-sharegpt-online-dcc06b18f3-51e577b17b`           | imported                    | 154.10271312217324 |   15490.6209520204 | 193.69931079949333 | 197.79772608540952 |
| `historical-pr-ascend-pr70-simllm-baseline-validation-sharegpt-online-ceec19abb0-312ca80a90`        | `historical-pr-ascend-pr70-simllm-baseline-validation-ttft-fix-sharegpt-online-ceec19abb0-312ca80a90`        | imported                    | 151.71956176581236 | 18389.533519154647 | 191.25389561507905 | 201.36698499089107 |
| `historical-pr-ascend-pr66-simllm-kv-manager-sharegpt-online-ceec19abb0-e0686f12d1`                 | `historical-pr-ascend-pr66-simllm-kv-manager-ttft-fix-sharegpt-online-ceec19abb0-e0686f12d1`                 | imported                    | 157.73268468915262 | 14902.951271270867 |  188.1373845431756 | 196.99755215784535 |
| `historical-pr-pr-77-ceec19abb0-instructcoder-online-ceec19abb0-51e577b17b`                         | `historical-pr-pr77-perfgate-l2-scenario-registry-ttft-fix-instructcoder-online-ceec19abb0-51e577b17b`       | imported                    | 180.25116377813768 |     45775.08857388 |  169.1315138575709 | 260.90687532041557 |
| `historical-pr-ascend-pr70-simllm-baseline-validation-instructcoder-online-ceec19abb0-312ca80a90`   | `historical-pr-ascend-pr70-simllm-baseline-validation-ttft-fix-instructcoder-online-ceec19abb0-312ca80a90`   | imported                    | 172.68390961208672 | 114943.90094966446 |  167.3632079936153 | 263.32407902691557 |
| `historical-pr-align-ascend-benchmark-perfgate-defaults-instructcoder-online-2fb7859dd0-51e577b17b` | `historical-pr-align-ascend-benchmark-perfgate-defaults-ttft-fix-instructcoder-online-2fb7859dd0-51e577b17b` | imported                    | 158.98391654052645 | 32166.155580103918 | 167.41515928739997 | 287.61788940119004 |
| `historical-pr-ascend-pr66-simllm-kv-manager-instructcoder-online-ceec19abb0-e0686f12d1`            | `historical-pr-ascend-pr66-simllm-kv-manager-instructcoder-online-ceec19abb0-e0686f12d1`                     | imported_after_second_retry | 164.64053460656092 | 159812.43235281965 |             166.37 |          299.42 ms |
