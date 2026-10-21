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
for every data file. Publication rejects uncommitted changes and unknown website data. The original
report generators must update these publication artifacts when publishing new measurements.

Files exceeding 1 MB are stored as deterministic gzip archives split into 900,000-byte parts. A
`.publication.json` descriptor records each part's hash; the publisher verifies and reconstructs the
exact original bytes. The website manifest records both the descriptor hash and the reconstructed
data hash.

The initial bootstrap is an exact archival import of public website data. Until its benchmark commit
is published, the manifest explicitly says `local-unpublished` and must not be presented as deployed
evidence.
