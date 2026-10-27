# Changelog

## 0.1.2

- Bind the ECPA activation manifest to the validated vLLM-HUST 0.31.1
  request-lifecycle-events v1 host line.

## [Unreleased]

### Added

- Added a vLLM-HUST request-lifecycle EventBus 1.0 sink for finished,
  preempted, and KV-reclaimed events, with callback-bound ECPA runtime
  evidence and optional trace export.
- Added a shared top-level `Makefile` with consistent `install-dev`, `smoke`, `test`, `lint`, `format`, `build`, `bench`, and `paper` targets.
- Added a top-level `CHANGELOG.md` so the template matches the standalone plugin repository structure used across the related vLLM and SGLang artifact repos.
- Added a `.benchmarks/` workspace plus a generic shared-workload smoke harness that consumes the repo-local case catalog from `llm-serving-workloads`.
- Added the standard template targets `shared-workloads-smoke` and `shared-workloads-test` so future derived repositories can expose a stable workload-driven full-test entrypoint.
- Added placeholder `paper/` workspace scaffolding so future standalone optimization repos can grow into a paper artifact without reshaping the root layout.
- Added a working paper scaffold under `paper/request_lifecycle_causal_profiler/` with a generic LaTeX draft, starter bibliography, and repo-local `tectonic` build Makefile.
- Added top-level `paper-pdf` and `paper-clean` targets so derived repositories start with a usable paper build entrypoint instead of a paper directory placeholder.
- Added a standard paper-side directory convention under the scaffolded paper workspace: `experiments/`, `results/`, `figures/`, and `tables/`.
- Added a generic `paper-assets` export path that refreshes paper-facing summaries from the paper `results/` directory before PDF compilation.
- Added `scripts/render_placeholder_paper_scaffold.py` as the reusable single-repo placeholder paper renderer, so workspace-level tooling can batch-apply the same scaffold without copying template text into umbrella scripts.

### Changed

- Promoted the ECPA Bundle to the narrow native observer carrier in 0.1.1;
  installation remains inert and historical KV-recovery tracing remains
  optional.
- Clarified changelog ownership: template changes must be recorded in this repository's own `CHANGELOG.md`, not in `/home/shuhao/sagellm/CHANGELOG.md`.
- Standardized local developer metadata by aligning `.[dev]` dependencies, pytest testpaths, and CONTRIBUTING workflow wording with the other standalone plugin repos.
- Changed the template's default environment guidance to use a dedicated `vllm-request-lifecycle-profiler-exp` clone of the shared vLLM baseline instead of reusing the shared env directly.
- Changed `install-dev` so the sibling `llm-serving-workloads` checkout is installed automatically when present.
- Changed the paper workspace from a documentation-only placeholder into a minimal artifact skeleton that derived repositories can compile immediately and then rename to the real paper topic.
- Changed the paper build contract to follow a stable `paper-assets -> pdf` flow so derived repositories can replace only the export logic while keeping the same top-level targets.
