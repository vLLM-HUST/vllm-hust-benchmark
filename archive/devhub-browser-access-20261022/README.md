# Direct browser access to existing Release members

These three small JSON files are byte-identical copies of members already archived in
`evidence-devhub-20261020-v1/devhub-evidence-20261020.tar.gz`. Their hashes and member paths match
`../dev-hub-20261020/manifest.json`; the downloaded archive was also SHA256-verified before reading.
The copies let website readers open the exact raw summary, adjudicated summary and adjudication
separately without downloading/extracting the whole archive. The Release remains authoritative.

No metric, adjudication decision or original byte was changed. The navigation builder has an
explicit `--extract-devhub` mode and refuses to replace existing copies.
