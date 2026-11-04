# Count-dataset-tokens Controls

Both original controls returned reward 0 with 0/1 tests passed. The oracle failed to retrieve
tokenizer configuration and had no local cache, before dataset loading or counting. This is a
dependency-transport failure, not an established counting or ARM64 defect. Original versions and
failures remain; no Agent admission or recovery score is added.

Run `python3 verify.py` for offline evidence checks. Published whitelist files are exact copies;
restricted original evidence is represented only by its SHA256. No experiment is run by this
verifier.
