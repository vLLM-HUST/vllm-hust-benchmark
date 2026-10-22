# GDN and OPset original evidence

These public-source snapshots preserve original bytes, source commits, Git blob identities, lengths
and SHA256 digests. GDN includes its evidence directory and root metadata/catalog; OPset includes
results, documentation and experiment instructions. This is custody only, not a new run or
performance qualification. Historical diagnostic, negative and eager-only evidence is not promoted
to formal leaderboard data. Original descriptions and source visibility are retained.

Source copies must remain until this archive merges, is downloaded back with every digest verified,
and consumer migrations pass all CI. Historical Git commits will not be rewritten. GDN's rejected
prototype patch is source code and must remain in its source repository; custody verification never
executes it.

## Completed consumer migration

[source-cleanup.json](source-cleanup.json) records both fully green merged MOD PRs, remote
source-tree verification, trusted default-branch checks and strict administrator-enforced GitHub
Actions protection. All 41 originals (including five native specification files) were downloaded
from the merged archive and verified before deleting 21 data/report copies. The rejected prototype
patch remains byte-identical in GDN. Documentation, metadata and native-spec measured sections now
point to originals; methods, admission limits and negative gates remain intact. GDN also gains
Python package and CPU native reference CI.

Website publication links for GDN and OPset now target these immutable original reports, with no
change to displayed measurements or the current report version.

[release-inventory.json](release-inventory.json) distinguishes reviewed Release assets from
completed migration: GDN and OPset have no Releases. Split-Batch has four Releases containing twelve
assets; the reviewed source distribution also embeds result/log files and remains follow-up debt,
not an exempt clean package. Website has a historical State Scheduler prerelease with six assets. No
Release asset was removed or republished; visibility and prerelease state were preserved. This batch
does not close issue 326.
