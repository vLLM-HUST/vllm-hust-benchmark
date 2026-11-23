# Website Publication Data

This directory is the canonical publication source for the website's `data/` files, including
measurements, dataset summaries, serving plans and reference prices. Original provenance embedded in
the artifacts is retained unchanged. Importing a file does not validate its measurement methodology
or turn an estimate into an observation.

Update data here, not in the website checkout. Publish only a committed source:

```sh
python scripts/sync_website_publication.py publish \
  --website /path/to/vllm-hust-website --revision <full-commit>
python scripts/sync_website_publication.py check \
  --website /path/to/vllm-hust-website --revision <full-commit>
```

The generated website manifest records the benchmark commit, repository path, SHA256 and byte count
for every data file. Publication checks the complete source path set and each file against the
requested commit before staging output. It rejects missing paths, unknown website data and multiple
source files that resolve to the same public path. Update these publication artifacts through the
original report generators when publishing new measurements.

Files exceeding 1 MB are stored as deterministic gzip archives split into 900,000-byte parts. A
`.publication.json` descriptor records each part's hash; the publisher verifies and reconstructs the
exact original bytes. The website manifest records both the descriptor hash and the reconstructed
data hash.

The initial bootstrap is an exact archival import of public website data. Until its benchmark commit
is published, the manifest explicitly says `local-unpublished` and must not be presented as deployed
evidence.

## Public metadata automation

Public GitHub/PyPI facts are refreshed by benchmark's `refresh-public-site-status.yml`. It preserves
curated acceptance evidence and proposes a benchmark data PR with regenerated navigation. The writer
uses the existing dataset credential so PR checks run; it never pushes main or automatically
approves runtime compatibility. Review and merge the source PR, then update the website publication
lock.

Website's version workflow renders only README from that reviewed lock and proposes a documentation
PR. It cannot commit website data. Benchmark snapshot previews are data-only artifacts under
`output/leaderboard-preview/`; they do not contain frozen leadership pins or form a deployable
website. Explicit frozen-target migrations retain the full target-object guard.

## Fixed B0 observations

`data/fixed-b0-original018-v1.json` independently preserves the published original018 B0: five fixed
profiles, four context/tenant phases, and original native compute/reference statistics. Regenerate
with `python scripts/build_fixed_baseline_observations.py`; `--check` compares every published byte
and the catalog descriptor against unchanged source reports. Candidate releases do not own or clear
this record. Displaying B0 alone never supplies a missing B1, a new pair, or a new acceptance
result. Keep the original model, workload, dates, source seals and reference denominator. Existing
paired reports retain their own admitted B0 observations.
