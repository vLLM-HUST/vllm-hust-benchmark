# Dataset matrix presentation correction

The published matrix retains task accuracy/success percentages in its existing quality table. SWE-smith-tranjactory replay appears separately as **output-throughput improvement +91.7%**, with its APC-on/512MiB KV profile and180s measurement window disclosed. It is not a task accuracy or complete-batch makespan result. Three paired repetitions and the independent execution-order repeat remain available; default display labels use human-readable repetitions rather than internal random-seed/run identifiers.

PR441 merged after all four CI checks passed. GitHub Pages reported built at the recorded merge. Actual Chrome checks passed at1440px Chinese,390px English and320px Chinese: five quality rows, one replay row, eight detailed records, filters/sort/evidence expansion, no JavaScript errors or viewport overflow. This receipt records UI verification; it does not assert a separate byte-for-byte HTTP fetch or add new model measurements. The underlying three-pair dataset is unchanged; see ../offline-burst-three-seed-20261017/.

The later matched MMLU-Pro/FrontierScience/HLE-Verified campaign is still running. No unfinished task scores are published as complete.
