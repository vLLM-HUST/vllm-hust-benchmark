"""How many tokens a request has: used for input length and output length alike.

Follows Llumnix's benchmark (llumnix-project/llumnix-ray, commit 3fb6c03,
`benchmark/benchmark_serving.py` lines 596-649, `gen_random_response_lens`).
The Llumnix paper (OSDI'24, section 6.1) calls the Zipf case "power-law length
distributions" with Short / Medium / Long means of 128 / 256 / 512 tokens.

Two distributions:

- uniform: every length in [minimum, maximum] equally likely.
- zipf:    a power law. Llumnix's procedure, step by step:
    1. ranks are 1, 2, ..., 2 * mean - 1;
    2. rank r is drawn with probability proportional to r ** -alpha
       (small lengths common, long lengths rare; larger alpha = shorter tail);
    3. every drawn length is multiplied by (mean / sample mean) and rounded up,
       so the sample mean equals `mean` whatever alpha is;
    4. lengths above `maximum` are cut to `maximum`.
  Llumnix picks alpha from (mean, maximum) with a hard-coded table
  (lines 616-625); here alpha is a parameter and that table is kept as
  LLUMNIX_ZIPF_ALPHA so the paper's settings can be reproduced by name.
"""

from __future__ import annotations

import math
import random

LENGTH_DISTRIBUTIONS = ("uniform", "zipf")

# Llumnix's table, benchmark_serving.py lines 616-625: (mean, maximum) -> alpha.
# Any other (mean, maximum) gets alpha = 1.0 in Llumnix.
LLUMNIX_ZIPF_ALPHA = {
    (1024, 6144): 1.0005,
    (512, 6144): 1.15,  # "Long" in the paper
    (256, 6144): 1.5,  # "Medium"
    (128, 6144): 2.0,  # "Short"
}


def uniform_lengths(
    random_generator: random.Random, count: int, minimum: int, maximum: int
) -> list[int]:
    return [random_generator.randint(minimum, maximum) for _ in range(count)]


def zipf_lengths(
    random_generator: random.Random,
    count: int,
    mean: int,
    maximum: int,
    alpha: float,
) -> list[int]:
    ranks = list(range(1, 2 * mean))  # step 1
    weights = [rank**-alpha for rank in ranks]  # step 2
    drawn = random_generator.choices(ranks, weights=weights, k=count)
    scale_to_mean = mean / (sum(drawn) / count)  # step 3
    return [min(math.ceil(length * scale_to_mean), maximum) for length in drawn]  # 4


def sample_lengths(
    random_generator: random.Random,
    count: int,
    distribution: str,
    *,
    minimum: int,
    maximum: int,
    mean: int,
    zipf_alpha: float,
) -> list[int]:
    if distribution == "uniform":
        return uniform_lengths(random_generator, count, minimum, maximum)
    if distribution == "zipf":
        return zipf_lengths(random_generator, count, mean, maximum, zipf_alpha)
    raise ValueError(
        f"unknown length distribution {distribution!r}; one of {LENGTH_DISTRIBUTIONS}"
    )
