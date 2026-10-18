"""When requests arrive: the time between two consecutive requests.

Follows Llumnix's benchmark (llumnix-project/llumnix-ray, commit 3fb6c03,
`benchmark/benchmark_serving.py` lines 39-48, `get_wait_time`), which the
Llumnix paper (OSDI'24, section 6.1) describes as "Poisson and Gamma
distributions with different request rates" with the coefficient of variation
(CV) "to adjust the burstiness". vLLM's `vllm bench serve --burstiness` is the
same Gamma family.

Three processes, all with mean inter-arrival time 1 / request_rate:

- constant: every gap is exactly 1 / request_rate (CV = 0).
- poisson:  gaps are exponential (CV = 1). A Gamma with CV = 1 is the same thing.
- gamma:    gaps are Gamma with the given CV. CV < 1 is smoother than Poisson,
            CV > 1 is burstier (many short gaps, a few very long ones).

For a Gamma with mean m and coefficient of variation c:
    variance = (c * m) ** 2
    shape    = m ** 2 / variance = 1 / c ** 2
    scale    = variance / m      = c ** 2 * m
which is exactly Llumnix's formula, rearranged.
"""

from __future__ import annotations

import random

ARRIVAL_PROCESSES = ("constant", "poisson", "gamma")


def inter_arrival_seconds(
    random_generator: random.Random,
    process: str,
    request_rate: float,
    coefficient_of_variation: float,
) -> float:
    """One gap, in seconds, between a request and the next one."""
    mean_gap_seconds = 1.0 / request_rate
    if process == "constant":
        return mean_gap_seconds
    if process == "poisson":
        return random_generator.expovariate(request_rate)
    if process == "gamma":
        if coefficient_of_variation <= 0:
            # a Gamma with no variance is the constant process; Llumnix would
            # divide by zero here
            return mean_gap_seconds
        shape = 1.0 / coefficient_of_variation**2
        scale = coefficient_of_variation**2 * mean_gap_seconds
        return random_generator.gammavariate(shape, scale)
    raise ValueError(f"unknown arrival process {process!r}; one of {ARRIVAL_PROCESSES}")


def arrival_times_seconds(
    random_generator: random.Random,
    count: int,
    process: str,
    request_rate: float,
    coefficient_of_variation: float,
) -> list[float]:
    """Arrival time of each request, the first one at 0 s (as in Llumnix)."""
    times = [0.0]
    for _ in range(count - 1):
        gap = inter_arrival_seconds(
            random_generator, process, request_rate, coefficient_of_variation
        )
        times.append(times[-1] + gap)
    return times
