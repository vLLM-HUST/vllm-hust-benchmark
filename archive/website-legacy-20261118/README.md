# Legacy website artifact custody

This is a byte-preserving Benchmark archive of historical website evidence, reports, screenshots and
root presentation attachments. `manifest.json` pins the source website commit, original paths,
Git blob IDs, SHA256 and byte counts. The archive uses deterministic gzip/tar split below 1 MB.

Reconstruct without changing the originals:

```sh
cat originals.tar.gz.part-* > /tmp/website-legacy-originals.tar.gz
python -m tarfile -l /tmp/website-legacy-originals.tar.gz
```

Run `python -m pytest tests/test_legacy_website_custody.py -q` from the repository root to verify every
part and archived file. Read individual members with Python `tarfile`; this archive is not executable.
Existing website paths remain compatibility copies. Retire them only after recording and validating
all consuming links under the website data policy. Current publication data remains separately owned
by `website-publication`; this archive does not publish or revise historical measurements.
