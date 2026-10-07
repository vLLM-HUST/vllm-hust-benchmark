# Issue 214 PPT-safe summary

Reference: vLLM-HUST @ 43341b177dbaa8c7f04662f71e885ee7dfe22704 + vLLM Ascend @ 0a46364814eedd3314f04eff3490c3ab422438bd (`43341b177dbaa8c7f04662f71e885ee7dfe22704` / `0a46364814eedd3314f04eff3490c3ab422438bd`)

Candidate: vLLM 0.18.0 @ bcf2be96120005e9aea171927f85055a6a5c0cf6 + vLLM Ascend 0.18.0 @ e18643f8a4d5bd9990727654318ad069ea0b56e2 (`bcf2be96120005e9aea171927f85055a6a5c0cf6` / `e18643f8a4d5bd9990727654318ad069ea0b56e2`)

Reference arrays are summary values. Candidate arrays are values read from validated repeat artifacts.

| Workload | Metric | Reference median (IQR) | Candidate median (IQR) | Delta | Evidence | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| agent-research-online | ttft_ms (ms) | 3111.066892 (160.019165) | 173.781067 (12097.679470) | N/A | B/B | ready |

Reference summary values: `[2867.210210853955,3187.2485411004163,3111.066892481176]`

Candidate exported values: `[24360.36418908043,173.78106713294983,165.0052485638298]`

Stability notice: High IQR requires cautious stability interpretation: candidate

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| instructcoder-online | ttft_ms (ms) | 114.606179 (1.009258) | 157.505901 (3.613228) | N/A | B/B | ready |

Reference summary values: `[114.60617898956116,115.27481575240017,113.25630023634403]`

Candidate exported values: `[164.17096158511413,157.50590062907577,156.94450654063985]`

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| prefix-repetition-online | ttft_ms (ms) | 392.090389 (32.996940) | 605.187046 (63.395742) | N/A | B/B | ready |

Reference summary values: `[427.2499468596652,384.06993267824873,393.3142483397387,360.70254458347335,427.7773219323717,390.8665301837027]`

Candidate exported values: `[548.0777548160404,674.8692390695214,605.1870459597558]`

Stability notice: High IQR requires cautious stability interpretation: candidate

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| random-latency | batch_latency_ms (ms) | 4913.778062 (7.848560) | 7188.682899 (132.182113) | N/A | B/B | ready |

Reference summary values: `[4904.096141954264,4913.77806176121,4919.793262224023]`

Candidate exported values: `[6999.5156718418,7188.682898692787,7263.87989812841]`

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| random-online | ttft_ms (ms) | 229.685024 (0.677828) | 254.217167 (2.580401) | N/A | B/B | ready |

Reference summary values: `[229.68502367148176,228.47401247592643,229.82966804178432]`

Candidate exported values: `[254.44302909076217,249.28222787566483,254.21716653741893]`

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| sharegpt-online | ttft_ms (ms) | 127.975247 (4.174979) | 152.446241 (0.492175) | N/A | B/B | ready |

Reference summary values: `[127.97524687135592,125.917899615597,134.26785849500448]`

Candidate exported values: `[152.5572005752474,151.57285124063492,152.4462412018329]`

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| sharegpt-throughput | throughput_tps (TPS) | 2048.152959 (13.679724) | 1686.877704 (26.417882) | N/A | B/B | ready |

Reference summary values: `[2066.3135875574776,2038.954139338247,2048.152959232967]`

Candidate exported values: `[1686.8777040113725,1691.5031160439933,1638.6673519911747]`

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| sonnet-throughput | throughput_tps (TPS) | 3614.039872 (17.797716) | 2837.543002 (12.035836) | N/A | B/B | ready |

Reference summary values: `[3626.749496459258,3591.1540650893967,3614.039871632891]`

Candidate exported values: `[2837.543001706054,2820.67128834616,2844.7429600836103]`

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified

| visionarena-online | ttft_ms (ms) | 437.949286 (22.610710) | 407.890600 (121.624838) | N/A | B/B | ready |

Reference summary values: `[440.5764604774304,448.3808410721831,456.2576252957806,435.32211213745177,419.9846773506142,419.30543774273247]`

Candidate exported values: `[647.0599274486303,407.89059953019023,403.81025174446404]`

Stability notice: High IQR requires cautious stability interpretation: candidate

Restrictions: input identity is not verified | reference raw evidence is not verified | candidate full raw archive is not verified
