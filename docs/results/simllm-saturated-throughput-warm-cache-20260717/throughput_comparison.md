# SimLLM saturated warm-cache throughput

| Metric | Baseline | SimLLM warm cache | Improvement |
| --- | ---: | ---: | ---: |
| Requests/s | 1.4054897373454687 | 4.789651013418685 | 240.78% |
| Output tokens/s | 44.975671595055 | 153.26883242939792 | 240.78% |
| Total tokens/s | 5801.861635762095 | 19771.679383392333 | 240.78% |

Measured with request_rate=inf, max_concurrency=16, completed=32.
