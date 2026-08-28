# Issue 135 input provenance

The evidence checksum manifest covers files committed in this report. The two external run inputs
are verified separately because the processed BurstGPT trace is about 31 MiB and the raw public
trace is about 221 MiB; neither belongs in Git history.

Run the verifier from a clean checkout:

```bash
python scripts/verify_issue_135_inputs.py
```

It downloads the immutable BurstGPT v2.0 release asset, verifies its GitHub-published SHA-256,
recreates the exact filtered/reordered CSV used by the benchmark, and verifies the resulting
SHA-256. Pass `--raw-csv PATH` to reuse an existing copy. Generated files go under the report's
ignored `inputs/` directory.

The small model-info cache entry used by the run is committed beside this document and is verified
by the same command. The repository copy adds the conventional final newline; the verifier removes
that one byte before checking the original cache-artifact hash. This file is evidence about the
recorded environment, not an input required to start a new benchmark process.

Expected identities:

- BurstGPT v2.0 `BurstGPT_3.csv`: `2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f`
- Processed BurstGPT CSV: `ef3bc195a041df6e35fd2f0572b93ed0c393482d3ec91b35e46c75bc409f6104`
- Qwen2 model-info cache entry: `87b61d3fbe4ccbceb12774955b252b11d1065437cd5edc57edf2058dd2f5f644`
