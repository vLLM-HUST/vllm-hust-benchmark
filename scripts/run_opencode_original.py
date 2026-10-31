"""Generate from original OpenCode input only; never execute code or teacher tests.

This is a published training-corpus workload, not an independent code-quality
benchmark. Explicit model/budget required; default is an offline plan.
"""

from run_openmath2_original import main


if __name__ == "__main__":
    main("OPENCODE")
